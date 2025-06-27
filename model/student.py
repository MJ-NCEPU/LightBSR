import torch
from torch import nn
import model.common as common
import torch.nn.functional as F
from timm.models.layers import DropPath


def make_model(args):
    return BlindSR(args)


class IDR_CB_S(nn.Module):
    def __init__(self):
        super(IDR_CB_S, self).__init__()
        self.sig1 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.GELU()
        )

    def forward(self, xk):
        return self.sig1(xk) + xk


class IDR_CB_C(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgafter = nn.Sequential(
            nn.Linear(48, 96),
            nn.GELU(),
            nn.Linear(96, 48),
            nn.GELU()
        )

    def forward(self, ca):
        return self.avgafter(ca) + ca


class channel_modulation(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super().__init__()

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.avgafter = nn.Sequential(
            nn.Linear(channels_in * 2, channels_in // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels_in // reduction, channels_out, bias=False),
        )

        self.sig = nn.Sigmoid()

    def forward(self, x, ca):
        ca_x = self.avg(x).squeeze(-1).squeeze(-1)
        ca = self.avgafter(torch.cat([ca_x, ca], dim=1))

        return x * self.sig(ca.unsqueeze(-1).unsqueeze(-1))


class Modulation(nn.Module):
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 5, 1, 2, bias=False)
        self.loc = channel_modulation(self.dim_untouched, self.dim_untouched, reduction=2)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.dim_conv3 + 8, self.dim_conv3, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.dim_conv3, 16, 3, 1, 1, bias=False)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x, ca, xk):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = x1 * self.sig(self.spatial_attention(torch.cat([xk, x1], dim=1)))
        x1 = self.partial_conv3(x1)
        x2 = self.loc(x2, ca)
        x = torch.cat((x1, x2), 1)
        return x


class IDR_AB(nn.Module):
    def __init__(self,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0,
                 layer_scale_init_value=1e-6
                 ):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Linear(dim, mlp_hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mlp_hidden_dim, dim)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.spatial_mixing = Modulation(dim, n_div)

        self.act = nn.GELU()
        self.norm = LayerNorm(dim, eps=1e-6)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x, ca, xk):
        shortcut = x
        x = self.spatial_mixing(x, ca, xk)
        x = self.act(x)

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = shortcut + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt_Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + x
        return x


class IDR_AG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_blocks):
        super(IDR_AG, self).__init__()
        self.n_blocks = n_blocks
        self.ca_change = IDR_CB_C()
        self.xk_change = IDR_CB_S()

        modules_body = [
            IDR_AB(n_feat) for _ in range(n_blocks)
        ]
        modules_body.append(ConvNeXt_Block(n_feat))
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x, ca, xk):
        res = x
        ca = self.ca_change(ca)
        xk = self.xk_change(xk)

        for i in range(self.n_blocks):
            res = self.body[i](res, ca, xk)
        res = self.body[-2](res)
        res = self.body[-1](res)
        res = res + x

        return res, ca, xk


def check_image_size(xk, x):
    _, _, H, W = xk.size()
    _, _, h, w = x.size()
    mod_pad_h = h - H
    mod_pad_w = w - W
    xk = F.pad(xk, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return xk


class LightBSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(LightBSR, self).__init__()

        self.n_groups = 8
        n_blocks = 8
        n_feats = 64
        kernel_size = 3
        scale = int(args.scale[0])

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(1.0, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(1.0, rgb_mean, rgb_std, 1)

        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # IDR-Converter
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.compress = nn.Sequential(
            nn.Linear(128, n_feats * 3 // 4, bias=False),
            nn.GELU()
        )
        # IDR-Converter
        self.pixshuffle = nn.PixelShuffle(4)
        self.afterpixshuffle = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.GELU()
        )

        # body
        modules_body = [
            IDR_AG(common.default_conv, n_feats, kernel_size, n_blocks) \
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, k_v):
        # sub mean
        x = self.sub_mean(x)
        # head
        x = self.head(x)

        ca = self.avg(k_v).squeeze(-1).squeeze(-1)
        ca = self.compress(ca)

        xk = self.pixshuffle(k_v)
        xk = check_image_size(xk, x)
        xk = self.afterpixshuffle(xk)

        # body
        res = x
        ca_ori = ca
        xk_ori = xk
        for i in range(self.n_groups):
            res, ca, xk = self.body[i](res, ca, xk)
        res = self.body[-1](res)
        res = res + x
        # tail
        x = self.tail(res)
        # add mean
        x = self.add_mean(x)

        return x, ca_ori, xk_ori


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x):
        fea = self.E(x)

        return fea


class BlindSR(nn.Module):
    def __init__(self, args):
        super(BlindSR, self).__init__()

        self.G = LightBSR(args)
        self.E = Encoder()

        if args.n_GPUs > 1:
            self.E = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.E)

    def forward(self, x):
        if self.training:
            fea = self.E(x)
            sr, ca, xk = self.G(x, fea)
            return sr, ca, xk
        else:
            fea = self.E(x)
            sr, ca, xk = self.G(x, fea)
            return sr


if __name__ == '__main__':
    model = BlindSR()
