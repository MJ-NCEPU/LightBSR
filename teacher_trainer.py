import os
import utility
import torch
from decimal import Decimal
from utils import util
import math
from utils.pca import cal_pca_matrix
from scipy.io import loadmat
import warnings

warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.loader = loader
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.noise = args.noise
        if args.n_GPUs > 1:
            self.model_E = torch.nn.parallel.DistributedDataParallel(self.model.get_model().E, [loader.local_rank])
        else:
            self.model_E = self.model.get_model().E

        pca_path = os.path.join(self.ckp.dir, 'PCA_matrix.mat')
        if os.path.exists(pca_path):
            self.pca = torch.from_numpy(loadmat(pca_path)['p']).cuda()
        else:
            self.pca = torch.from_numpy(
                cal_pca_matrix(path=pca_path, ksize=21, l_max=10.0, dim_pca=15, num_samples=5000)).cuda()

        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if args.n_GPUs > 1:
            if self.args.resume != 0 and not args.test_only:
                self.optimizer.load_state_dict(
                    torch.load(os.path.join(self.ckp.dir, 'opt', 'opt_{}.pt'.format(self.args.resume)),
                               map_location='cuda:{}'.format(args.local_rank))
                )
                for _ in range(self.args.resume): self.scheduler.step()
        else:
            if self.args.resume != 0 and not args.test_only:
                self.optimizer.load_state_dict(
                    torch.load(os.path.join(self.ckp.dir, 'opt', 'opt_{}.pt'.format(self.args.resume)))
                )
                for _ in range(self.args.resume): self.scheduler.step()

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1

        if epoch <= self.args.epochs_encoder:
            lr = self.args.lr_encoder * (self.args.gamma_encoder ** (epoch // self.args.lr_decay_encoder))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            if epoch > self.args.epochs_encoder and epoch <= (self.args.epochs_encoder + 5):
                lr = self.args.max_lr_sr * (epoch - self.args.epochs_encoder) / 5
            else:
                lr = self.args.min_lr_sr + (self.args.max_lr_sr - self.args.min_lr_sr) * \
                     (1 + math.cos((epoch - (self.args.epochs_encoder + 5)) / (self.args.epochs_sr - 5) * math.pi)) / 2
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()

        degrade = util.SRMDPreprocessing(
            self.scale[0],
            kernel_size=self.args.blur_kernel,
            blur_type=self.args.blur_type,
            sig_min=self.args.sig_min,
            sig_max=self.args.sig_max,
            lambda_min=self.args.lambda_min,
            lambda_max=self.args.lambda_max,
            noise=self.args.noise
        )

        timer = utility.timer()
        losses_contrast, losses_sr = utility.AverageMeter(), utility.AverageMeter()
        for batch, (hr, _, idx_scale) in enumerate(self.loader_train):
            hr = hr.cuda()  # b, n, c, h, w

            lr_complex, b_kernels, noise = degrade(hr)  # b, n, c, h, w
            b, n, c, h, w = lr_complex.shape

            # TODO: setting1
            degradation_vector = torch.mm(b_kernels.view(b, -1), self.pca).view(b, 1, -1, 1, 1).float()
            degradation_map = degradation_vector.repeat(1, n, 1, h, w)
            # TODO: setting2
            # noise = noise.repeat(1, n, 3, h, w)
            # degradation_vector = torch.mm(b_kernels.view(b, -1), self.pca).view(b, 1, -1, 1, 1).float()
            # degradation_map = torch.cat((degradation_vector.repeat(1, n, 1, h, w), noise), dim=2)

            self.optimizer.zero_grad()

            timer.tic()
            if epoch <= self.args.epochs_encoder:
                encoder_input = torch.cat((lr_complex, degradation_map), dim=2)
                _, loss_constrast = self.model_E(encoder_input)

                loss = loss_constrast
                losses_contrast.update(loss_constrast.item())
            else:
                sr, loss_constrast = self.model((lr_complex, degradation_map))
                loss_SR = self.loss(sr * 255., hr[:, 0, ...])

                loss = loss_constrast + loss_SR
                losses_sr.update(loss_SR.item())
                losses_contrast.update(loss_constrast.item())

            loss.backward()
            self.optimizer.step()
            timer.hold()

            if epoch <= self.args.epochs_encoder:
                if self.args.n_GPUs > 1:
                    if self.loader.rank == 0:
                        if (batch + 1) % self.args.print_every == 0:
                            self.ckp.write_log(
                                'Epoch: [{:03d}][{:04d}/{:04d}]\t'
                                'Loss [contrastive loss: {:.3f}]\t'
                                'Time [{:.1f}s]'.format(
                                    epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                                    losses_contrast.avg,
                                    timer.release()
                                ))
                else:
                    if (batch + 1) % self.args.print_every == 0:
                        self.ckp.write_log(
                            'Epoch: [{:03d}][{:04d}/{:04d}]\t'
                            'Loss [contrastive loss: {:.3f}]\t'
                            'Time [{:.1f}s]'.format(
                                epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                                losses_contrast.avg,
                                timer.release()
                            ))
            else:
                if self.args.n_GPUs > 1:
                    if self.loader.rank == 0:
                        if (batch + 1) % self.args.print_every == 0:
                            self.ckp.write_log(
                                'Epoch: [{:04d}][{:04d}/{:04d}]\t'
                                'Loss [SR loss:{:.3f} | contrastive loss: {:.3f}]\t'
                                'Time [{:.1f}s]'.format(
                                    epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                                    losses_sr.avg, losses_contrast.avg,
                                    timer.release(),
                                ))
                else:
                    if (batch + 1) % self.args.print_every == 0:
                        self.ckp.write_log(
                            'Epoch: [{:04d}][{:04d}/{:04d}]\t'
                            'Loss [SR loss:{:.3f} | contrastive loss: {:.3f}]\t'
                            'Time [{:.1f}s]'.format(
                                epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                                losses_sr.avg, losses_contrast.avg,
                                timer.release(),
                            ))

        self.loss.end_log(len(self.loader_train))

        if self.args.n_GPUs > 1:
            if self.loader.rank == 0:
                print(f"save: rank = {self.loader.rank}")
                target = self.model.get_model()
                model_dict = target.state_dict()
                torch.save(
                    model_dict,
                    os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch))
                )
                torch.save(
                    self.optimizer.state_dict(),
                    os.path.join(self.ckp.dir, 'opt', 'opt_{}.pt'.format(epoch))
                )
        else:
            target = self.model.get_model()
            model_dict = target.state_dict()
            torch.save(
                model_dict,
                os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch))
            )
            torch.save(
                self.optimizer.state_dict(),
                os.path.join(self.ckp.dir, 'opt', 'opt_{}.pt'.format(epoch))
            )

    def test(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()

        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.loader_test.dataset.set_scale(idx_scale)
                eval_psnr = 0
                eval_ssim = 0

                degrade = util.SRMDPreprocessing(
                    self.scale[0],
                    kernel_size=self.args.blur_kernel,
                    blur_type=self.args.blur_type,
                    sig=self.args.sig,
                    lambda_1=self.args.lambda_1,
                    lambda_2=self.args.lambda_2,
                    theta=self.args.theta,
                    noise=self.args.noise
                )

                for idx_img, (hr, filename, _) in enumerate(self.loader_test):
                    hr = hr.cuda()  # b, 1, c, h, w
                    hr = self.crop_border(hr, scale)
                    lr, blur, noise_l = degrade(hr, random=False)
                    hr = hr[:, 0, ...]  # b, c, h, w
                    lr = lr[:, 0, ...]  # b, c, h, w

                    b, c, h, w = lr.shape
                    # TODO: setting1
                    degradation_vector = torch.mm(blur.view(b, -1), self.pca).view(b, -1, 1, 1).float()
                    degradation_map = degradation_vector.repeat(1, 1, h, w)
                    # TODO: setting2
                    # noise_l = noise_l[:, 0, ...].repeat(1, 3, h, w)  # b 1 1 1 --> b 1 h w
                    # degradation_vector = torch.mm(blur.view(b, -1), self.pca).view(b, -1, 1, 1).float()
                    # degradation_map = torch.cat((degradation_vector.repeat(1, 1, h, w), noise_l), dim=1)

                    # inference
                    timer_test.tic()
                    sr, _, _ = self.model((lr, degradation_map))
                    sr *= 255.0
                    timer_test.hold()

                    sr = utility.quantize(sr, self.args.rgb_range)
                    hr = utility.quantize(hr, self.args.rgb_range)

                    # metrics
                    eval_psnr += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    eval_ssim += utility.calc_ssim(
                        sr, hr, scale,
                        benchmark=self.loader_test.dataset.benchmark
                    )

                    # save results
                    if self.args.save_results:
                        save_list = [sr]
                        filename = filename[0]
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_test)
                self.ckp.write_log(
                    '[Epoch {}---{} x{} noise {}]\tPSNR: {:.3f} SSIM: {:.4f} Time: {:.10f}s'.format(
                        self.args.resume,
                        self.args.data_test,
                        scale,
                        self.noise,
                        eval_psnr / len(self.loader_test),
                        eval_ssim / len(self.loader_test),
                        timer_test.release(),
                    ))

    def crop_border(self, img_hr, scale):
        b, n, c, h, w = img_hr.size()

        img_hr = img_hr[:, :, :, :int(h // scale * scale), :int(w // scale * scale)]

        return img_hr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs_encoder + self.args.epochs_sr
