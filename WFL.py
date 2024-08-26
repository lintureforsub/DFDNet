"""
note: version adaptation for PyTorch > 1.7.1
date: 16th August 2024
"""
import torch
import torch.nn as nn
import torch.fft
import numpy as np


class WeightedFL(nn.Module):

    def __init__(self, train, T, H, W, log_matrix=True):
        super(WeightedFL, self).__init__()
        self.log_matrix = log_matrix
        self.train = train
        mean = [T // 2, H // 2, W // 2]
        std = [8, 12, 12] # 
        self.metrix = torch.tensor(1.5 - self.gaussian_3d(T, H, W, mean, std)).unsqueeze(1).unsqueeze(0).unsqueeze(0).to('cuda')

    def data2spec(self, x):
        """
        three times fft to calculate 3D fft
        torch.fft is only available after PyTorch version 1.7.1
        if your Pytorch version < 1.7.1, please use the rfft
        """
        freq = torch.fft.fft(torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2),dim=-3)
        #freq = torch.fft.fftshift(freq)
        #  freq = torch.rfft(y, 3, onesided=False, normalized=True)  # if PyTorch < 1.7.1
        freq = torch.stack([freq.real, freq.imag], -1)

        return freq.unsqueeze(1)

    def gaussian_3d(self, T, H, W, mean, std):
        t, h, w = np.meshgrid(np.arange(T), np.arange(H), np.arange(W), indexing='ij')
        dist = ((t - mean[0]) / std[0]) ** 2 + ((h - mean[1]) / std[1]) ** 2 + ((w - mean[2]) / std[2]) ** 2
        gauss = np.exp(-0.5 * dist)
        return gauss
    def loss_formulation(self, recon_freq, real_freq):

        dis = recon_freq - real_freq
        loss = dis[..., 0]**2 + dis[..., 1]**2
        loss = loss * self.metrix.expand_as(loss)
        if self.train:
            return torch.mean(loss)
        else:
            return torch.mean(loss).cpu().detach().numpy()

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
        Out:
            numpy ndarray
        Example:
            # definition
            weighted_floss = ThreeDFL()
            # calculate
            pred = network(input)
            loss = weighted_floss(pred, target)
        """
        # calculate
        pred_freq = self.data2spec(pred)
        target_freq = self.data2spec(target)

        # calculate loss
        return self.loss_formulation(pred_freq, target_freq)


