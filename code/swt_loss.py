import torch
import torch.nn as nn
import SWT
import pywt
import numpy as np



def soft_threshold(x, t):
    return torch.sign(x) * torch.relu(torch.abs(x) - t)


class SWTLoss(nn.Module):
    def __init__(self, loss_weight_ll=0.02, loss_weight_lh=0.02, loss_weight_hl=0.02, loss_weight_hh=0.02, reduction='mean'):
        super(SWTLoss, self).__init__()
        self.loss_weight_ll = loss_weight_ll
        self.loss_weight_lh = loss_weight_lh
        self.loss_weight_hl = loss_weight_hl
        self.loss_weight_hh = loss_weight_hh

        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        wavelet = pywt.Wavelet('sym19')

        dlo = wavelet.dec_lo
        an_lo = np.divide(dlo, sum(dlo))
        an_hi = wavelet.dec_hi
        rlo = wavelet.rec_lo
        syn_lo = 2 * np.divide(rlo, sum(rlo))
        syn_hi = wavelet.rec_hi

        filters = pywt.Wavelet('wavelet_normalized', [an_lo, an_hi, syn_lo, syn_hi])
        sfm = SWT.SWTForward(1, filters, 'periodic').to("cuda")

        pred = torch.tensor(pred, dtype=torch.float32, device='cuda')
        target = torch.tensor(target, dtype=torch.float32, device='cuda')


        # 对CT图像进行小波变换
        wavelet_pred = sfm(pred)[0]
        wavelet_target = sfm(target)[0]

        LL_pred = wavelet_pred[:, 0:1, :, :]
        LH_pred = wavelet_pred[:, 1:2, :, :]
        HL_pred = wavelet_pred[:, 2:3, :, :]
        HH_pred = wavelet_pred[:, 3:, :, :]

        LL_target = wavelet_target[:, 0:1, :, :]
        LH_target = wavelet_target[:, 1:2, :, :]
        HL_target = wavelet_target[:, 2:3, :, :]
        HH_target = wavelet_target[:, 3:, :, :]



        # 计算损失
        loss_LL = self.loss_weight_ll * self.criterion(LL_pred, LL_target)
        loss_LH = self.loss_weight_lh * self.criterion(LH_pred, LH_target)
        loss_HL = self.loss_weight_hl * self.criterion(HL_pred, HL_target)
        loss_HH = self.loss_weight_hh * self.criterion(HH_pred, HH_target)

        return loss_LL + loss_LH + loss_HL + loss_HH