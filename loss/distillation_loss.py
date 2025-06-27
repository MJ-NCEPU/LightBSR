import torch.nn as nn
import torch.nn.functional as F


class KL_L1(nn.Module):
    def __init__(self, temperature=0.15):
        super(KL_L1, self).__init__()
        self.temperature = temperature

    def forward(self, g_s, g_t):
        '''
        g_s B 128
        g_t B 128
        '''
        loss = 0
        for i in range(len(g_s)):
            student_distance = F.log_softmax(g_s[i] / self.temperature, dim=1)
            teacher_distance = F.softmax(g_t[i].detach() / self.temperature, dim=1)
            loss_distill_dis = F.kl_div(student_distance, teacher_distance, reduction='batchmean')
            labs = nn.L1Loss()(g_s[i], g_t[i].detach())
            loss = loss + loss_distill_dis + 0.1 * labs
        return loss


class Fea_loss(nn.Module):
    def __init__(self):
        super(Fea_loss, self).__init__()
        self.criterion = nn.MSELoss(reduce=True, size_average=True)

    def forward(self, g_s, g_t):
        loss = self.criterion(g_s, g_t.detach())
        return loss
