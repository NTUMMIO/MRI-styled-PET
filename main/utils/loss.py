import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_msssim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)
        self.kernel_x = self.kernel_x.cuda()
        self.kernel_y = self.kernel_y.cuda()
    def forward(self, outputs, targets):
        gx_target = F.conv2d(targets, self.kernel_x, stride=1, padding=1)
        gy_target = F.conv2d(targets, self.kernel_y, stride=1, padding=1)
        gx_output = F.conv2d(outputs, self.kernel_x, stride=1, padding=1)
        gy_output = F.conv2d(outputs, self.kernel_y, stride=1, padding=1)
        grad_loss = F.mse_loss(gx_target, gx_output) + F.mse_loss(gy_target, gy_output)
        return grad_loss

def Fro_LOSS(batchimg):
    
    fro_norm = torch.norm(batchimg, p= 'fro',dim=(2,3))/ (int(batchimg.shape[2]) * int(batchimg.shape[3]))
    return fro_norm
def features_grad(features):
    kernel = torch.FloatTensor([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
    kernel = kernel[None,None,:,:].cuda()
    print("features",features.shape)
    c, _, _ = features.shape
    c = int(c)
    for i in range(c):
        feat=features[i, :, :]
        feat=feat[None,None,:,:]
        print("feat", feat.shape)
        fg = F.conv2d(feat, kernel,padding = 1)
        if i == 0:
            fgs = fg
        else:
            fgs = torch.cat([fgs, fg], axis = -1)
    return fgs


class AdaptiveMSE(nn.Module):
    def __init__(self):
        super(AdaptiveMSE, self).__init__()

    def forward(self, source1, source2, output):
        
        mse1 = Fro_LOSS(output-source1)
        mse2 = Fro_LOSS(output-source2)
        for i in range(len(source1)):
            self.m1 = torch.mean(torch.square(source1[i]), axis = [1, 2])
            self.m2 = torch.mean(torch.square(source2[i]), axis = [1, 2])

            if i == 0:

                self.ws1 = self.m1[:,None]
                self.ws2 = self.m2[:,None]
            else:

                self.ws1 = torch.cat([self.ws1,self.m1[:,None]], axis = -1)
                self.ws2 = torch.cat([self.ws2,self.m2[:,None]], axis = -1)


        self.s1 = torch.mean(self.ws1, axis = -1) / len(source1)
        self.s2 = torch.mean(self.ws2, axis = -1) / len(source2)

        self.s = F.softmax((torch.cat([self.s1[:,None], self.s2[:,None]], axis = -1)))

        self.mse_loss1=torch.mean(self.s[:, 0] * mse1)
        self.mse_loss2=torch.mean(self.s[:, 1] * mse2)

        return self.mse_loss1,self.mse_loss2


class AdaptiveSSIM(nn.Module):
    def __init__(self):
        super(AdaptiveSSIM, self).__init__()

    def forward(self, source1, source2, output):
        ssim1 = 1-pytorch_msssim.ssim(output, source1, size_average=False, val_range=1.0)
        ssim2 = 1-pytorch_msssim.ssim(output, source2, size_average=False, val_range=1.0)

        for i in range(len(source1)):
            self.m1 = torch.mean(torch.square(source1[i]), axis = [1, 2])
            self.m2 = torch.mean(torch.square(source2[i]), axis = [1, 2])

            if i == 0:
                self.ws1 = self.m1[:,None]
                self.ws2 = self.m2[:,None]
            else:
                self.ws1 = torch.cat([self.ws1,self.m1[:,None]], axis = -1)
                self.ws2 = torch.cat([self.ws2,self.m2[:,None]], axis = -1)

        self.s1 = torch.mean(self.ws1, axis = -1) / len(source1)
        self.s2 = torch.mean(self.ws2, axis = -1) / len(source2)
        self.s = F.softmax((torch.cat([self.s1[:,None], self.s2[:,None]], axis = -1)))
        self.ssim_loss1=torch.mean(self.s[:, 0] * ssim1)
        self.ssim_loss2=torch.mean(self.s[:, 1] * ssim2)
        
        return self.ssim_loss1,self.ssim_loss2    
    

#================================================================================
# Adapted from pytorch-ssim
# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
#================================================================================

import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: Tensor, win: Tensor) -> Tensor:
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    size_average: bool = True,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
) -> Tuple[Tensor, Tensor]:
    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
) -> Tensor:
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)

def _scalabel_ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    alpha: float = 1,
    
    
) -> Tuple[Tensor, Tensor]:
    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)
    # sigma1_sigma1_2 =  torch.power(sigma1_sq, 1/2)* torch.power(sigma2_sq, 1/2)
    
    # sigma1_sq_complex = torch.complex(sigma1_sq, torch.zeros_like(sigma1_sq))
    # sigma1 = torch.sqrt(sigma1_sq_complex)
    
    # sigma2_sq_complex = torch.complex(sigma2_sq, torch.zeros_like(sigma2_sq))
    # sigma2 = torch.sqrt(sigma2_sq_complex)
    
    # luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    # contrast = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    # structure = (sigma12 + C2 / 2) / (sigma1 * sigma2 + C2 / 2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2) 
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1))**alpha * cs_map
    # ssim_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    
    # ssim_map =  torch.pow(luminance, alpha) * torch.pow(contrast, beta) * torch.pow(structure, gamma)# * * 

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    # cs = torch.flatten(cs_map, 2).mean(-1)
    # print(ssim_per_channel.shape, cs.shape)
    return ssim_per_channel#, cs

class AdaptiveMSSSIM(nn.Module):
    def __init__(self,
        data_range: float = 255,
        size_average: bool = True,
        nonnegative_ssim: bool = False,
        noncomplex_ssim: bool = False,
        
    ):
        super(AdaptiveMSSSIM, self).__init__()
        
        self.data_range=data_range
        self.size_average=size_average
        self.win_sigma=1.5
        self.nonnegative_ssim=nonnegative_ssim
        self.noncomplex_ssim=noncomplex_ssim


    def forward(self, X, Y, sets_winSize_weighting = [(11, 1.0)]):
        self.msssim=0
        if not X.shape == Y.shape:
            raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

        for d in range(len(X.shape) - 1, 1, -1):
            X = X.squeeze(dim=d)
            Y = Y.squeeze(dim=d)

        if len(X.shape) not in (4, 5):
            raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")
        # print(self.sets_win_size_weighting)
        for (current_winsize, current_weighting) in sets_winSize_weighting:
            
            if not (current_winsize % 2 == 1):
                raise ValueError("Window size should be odd.")

            win = _fspecial_gauss_1d(current_winsize, self.win_sigma)
            win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

            ssim_per_channel = _scalabel_ssim(X, Y, data_range=self.data_range, win=win)
            if self.nonnegative_ssim:
                ssim_per_channel = torch.relu(ssim_per_channel)
            if self.noncomplex_ssim:
                ssim_per_channel = torch.real(ssim_per_channel)
            if self.size_average:
                ssim_per_channel = ssim_per_channel.mean()
            else:
                ssim_per_channel = ssim_per_channel.mean(1)

            if self.msssim==0:
                self.msssim=(1-ssim_per_channel)*current_weighting
            else:
                self.msssim=self.msssim+(1-ssim_per_channel)*current_weighting

        return self.msssim


def scalable_ssim(X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
    noncomplex_ssim: bool = True,
    alpha: float = 1
) -> Tensor:
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel = _scalabel_ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K, alpha=alpha)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)
    if noncomplex_ssim:
        ssim_per_channel = torch.real(ssim_per_channel)
    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)

def ms_ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    weights: Optional[List[float]] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
) -> Tensor:
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights_tensor = X.new_tensor(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights_tensor.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # type: ignore  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights_tensor.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)

def scalable_ms_ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    weights: Optional[List[float]] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    alpha: float = 1
) -> Tensor:
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights_tensor = X.new_tensor(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights_tensor.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _scalabel_ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K, alpha=alpha)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.real(ssim_per_channel)
    ssim_per_channel = torch.relu(ssim_per_channel)  # type: ignore  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights_tensor.view(-1, 1, 1), dim=0)


        
    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)
    
class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
        nonnegative_ssim: bool = False,
    ) -> None:
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        weights: Optional[List[float]] = None,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    ) -> None:
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        )
        
        
# if __name__ == "__main__":
#     mri=torch.from_numpy(np.array(Image.open("/home/linyunong/project/style_transfer/output_images/src/fold0/MRI/AD_FDG_250/002_S_1268.nii/002_S_1268_Axial_40.tif")))
#     pet=torch.from_numpy(np.array(Image.open("/home/linyunong/project/style_transfer/output_images/src/fold0/PET/AD_FDG_250/002_S_1268.nii/002_S_1268_Axial_40.tif")))
#     outputs=torch.from_numpy(np.array(Image.open("/home/linyunong/project/style_transfer/output_images/model_checkpoint/sc_att_gm/fold0/AD_FDG_250/002_S_1268.nii/002_S_1268_Axial_40.tif")))
#     mri=mri[None, None, :, :]
#     pet=pet[None, None, :, :]
#     outputs=outputs[None, None, :, :]
    

#     scalable_ssim_FDG = 1 - scalable_ssim(outputs, pet, data_range=1.0, size_average=True, alpha=0.5)#, beta=0.5, gamma=0.25)
#     scalable_ssim_MRI = 1 - scalable_ssim(outputs, mri, data_range=1.0, size_average=True, alpha=0.5)#, beta=0.5, gamma=0.25)
#     scalable_ssim_loss = 1 - scalable_ssim(pet, mri, data_range=1.0, size_average=True, alpha=0.5)#, beta=0.5, gamma=0.25)
#     print("with pet:", scalable_ssim_FDG, "with mri:", scalable_ssim_MRI, "between:",  scalable_ssim_loss)

#     ssim_loss_FDG = 1 - ssim(outputs, pet, data_range=1.0, size_average=True)
#     ssim_loss_MRI = 1 - ssim(outputs, mri, data_range=1.0, size_average=True)
#     ssim_loss = 1 - ssim(pet, mri, data_range=1.0, size_average=True)
#     print("with pet:", ssim_loss_FDG, "with mri:",  ssim_loss_MRI, "between:",  ssim_loss)




class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, MRI_mask, PET_pred):

        # Calculate gradients
        gradients = torch.abs(MRI_mask - PET_pred)
        # print(gradients.device)
        # Calculate boundary weights
        boundary_weights = torch.maximum(
            torch.where((MRI_mask == 0.25).to(MRI_mask.device) & self.check_adjacent_positions(MRI_mask, 1).to(MRI_mask.device), torch.tensor(0.75).to(MRI_mask.device), torch.tensor(0.0).to(MRI_mask.device)),
            torch.where((MRI_mask == 0.0).to(MRI_mask.device) & self.check_adjacent_positions(MRI_mask, 0.25).to(MRI_mask.device), torch.tensor(0.25).to(MRI_mask.device), torch.tensor(0.0).to(MRI_mask.device))
        )
        # print(boundary_weights.device)

        boundary_weights = torch.maximum(boundary_weights,
                                         torch.where((MRI_mask == 0.0).to(MRI_mask.device) & self.check_adjacent_positions(MRI_mask, 1).to(MRI_mask.device), torch.tensor(1.0).to(MRI_mask.device), torch.tensor(0.0).to(MRI_mask.device)))
        # print(boundary_weights.shape)
        weighted_gradients = gradients * boundary_weights

        # Combine MSE loss and weighted boundary gradients
        total_loss = torch.mean(weighted_gradients)

        return total_loss

    def pad_outer(self, tensor, padding_size):
        return torch.nn.functional.pad(tensor, (padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)

    def check_adjacent_positions(self, tensor, value):
        # Roll tensor in all four directions
        up = torch.roll(tensor, shifts=(-1, 0), dims=(2, 3))
        down = torch.roll(tensor, shifts=(1, 0), dims=(2, 3))
        left = torch.roll(tensor, shifts=(0, -1), dims=(2, 3))
        right = torch.roll(tensor, shifts=(0, 1), dims=(2, 3))
        
        # Compare adjacent positions with the specified value
        adjacent_positions = ((up == value) | (down == value) | (left == value) | (right == value))
        # print(value, adjacent_positions.device)
        return adjacent_positions
    
class RelativeBoundaryLoss(nn.Module):
    def __init__(self):
        super(RelativeBoundaryLoss, self).__init__()

    def forward(self, MRI_mask, PET_pred):
        MRI_mask_scaled=self.linear_scale(MRI_mask, torch.min(PET_pred), torch.max(PET_pred))
        # Calculate gradients
        gradients = torch.abs(MRI_mask_scaled - PET_pred)
        # print(gradients.device)
        # Calculate boundary weights
        
        # ====================
        # Experiment 1
        # boundary_weights = torch.maximum(
        #     torch.where((MRI_mask == 0.25).to(MRI_mask.device) & self.check_adjacent_positions(MRI_mask, 1).to(MRI_mask.device), torch.tensor(0.75).to(MRI_mask.device), torch.tensor(0.0).to(MRI_mask.device)),
        #     torch.where((MRI_mask == 0.0).to(MRI_mask.device) & self.check_adjacent_positions(MRI_mask, 0.25).to(MRI_mask.device), torch.tensor(0.25).to(MRI_mask.device), torch.tensor(0.0).to(MRI_mask.device))
        # )
        # print(boundary_weights)

        # boundary_weights = torch.maximum(boundary_weights,
        #                                  torch.where((MRI_mask == 0.0).to(MRI_mask.device) & self.check_adjacent_positions(MRI_mask, 1).to(MRI_mask.device), torch.tensor(1.0).to(MRI_mask.device), torch.tensor(0.0).to(MRI_mask.device)))
        # print(boundary_weights)
        # ====================
        # Experiment 2
        boundary_weights = MRI_mask
        # ====================
        
        weighted_gradients = gradients * boundary_weights

        # Combine MSE loss and weighted boundary gradients
        total_loss = torch.mean(weighted_gradients)

        return total_loss

    def pad_outer(self, tensor, padding_size):
        return torch.nn.functional.pad(tensor, (padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)

    def check_adjacent_positions(self, tensor, value):
        # Roll tensor in all four directions
        up = torch.roll(tensor, shifts=(-1, 0), dims=(2, 3))
        down = torch.roll(tensor, shifts=(1, 0), dims=(2, 3))
        left = torch.roll(tensor, shifts=(0, -1), dims=(2, 3))
        right = torch.roll(tensor, shifts=(0, 1), dims=(2, 3))
        
        # upper_left = torch.roll(torch.roll(tensor, shifts=(-1, -1), dims=(2, 3)), shifts=(-1, 0), dims=(2, 3))
        # upper_right = torch.roll(torch.roll(tensor, shifts=(-1, 1), dims=(2, 3)), shifts=(-1, 0), dims=(2, 3))
        # lower_left = torch.roll(torch.roll(tensor, shifts=(1, -1), dims=(2, 3)), shifts=(1, 0), dims=(2, 3))
        # lower_right = torch.roll(torch.roll(tensor, shifts=(1, 1), dims=(2, 3)), shifts=(1, 0), dims=(2, 3))
        
        # Compare adjacent positions with the specified value
        adjacent_positions = ((up == value) | (down == value) | (left == value) | (right == value))# | (upper_left == value) | (upper_right == value) | (lower_left == value) | (lower_right == value))
        # print(value, adjacent_positions.device)
        return adjacent_positions
    def linear_scale(self, tensor, min_value, max_value):
        # Calculate the current min and max values of the tensor
        current_min = torch.min(tensor)
        current_max = torch.max(tensor)
        
        # Scale the tensor linearly to the specified range
        scaled_tensor = (tensor - current_min) * (max_value - min_value) / (current_max - current_min) + min_value
        
        return scaled_tensor
    
class BoundaryGradient(nn.Module):
    def __init__(self):
        super(BoundaryGradient, self).__init__()

    def forward(self, PET_pred, PET_input, MRI_mask, weighted_mask):
        
        #Linear transform MRI_mask to relative intensity
        MRI_mask_scaled=self.linear_scale(MRI_mask, torch.min(PET_pred), torch.max(PET_pred))
        
        # Calculate MAE
        MRI_gradients = torch.abs(MRI_mask_scaled - PET_pred)
        PET_gradients = torch.abs(PET_input - PET_pred)
        
        #Weighting based on tissue type
        MRI_weighted_gradients = MRI_gradients * weighted_mask
        PET_weighted_gradients = PET_gradients * weighted_mask
        
        #Average
        MRI_boundary_gradient = torch.mean(MRI_weighted_gradients)
        PET_boundary_gradient = torch.mean(PET_weighted_gradients)
        
        return MRI_boundary_gradient, PET_boundary_gradient

    def pad_outer(self, tensor, padding_size):
        return torch.nn.functional.pad(tensor, (padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)

    def check_adjacent_positions(self, tensor, value):
        # Roll tensor in all four directions
        up = torch.roll(tensor, shifts=(-1, 0), dims=(2, 3))
        down = torch.roll(tensor, shifts=(1, 0), dims=(2, 3))
        left = torch.roll(tensor, shifts=(0, -1), dims=(2, 3))
        right = torch.roll(tensor, shifts=(0, 1), dims=(2, 3))
        
        # upper_left = torch.roll(torch.roll(tensor, shifts=(-1, -1), dims=(2, 3)), shifts=(-1, 0), dims=(2, 3))
        # upper_right = torch.roll(torch.roll(tensor, shifts=(-1, 1), dims=(2, 3)), shifts=(-1, 0), dims=(2, 3))
        # lower_left = torch.roll(torch.roll(tensor, shifts=(1, -1), dims=(2, 3)), shifts=(1, 0), dims=(2, 3))
        # lower_right = torch.roll(torch.roll(tensor, shifts=(1, 1), dims=(2, 3)), shifts=(1, 0), dims=(2, 3))
        
        # Compare adjacent positions with the specified value
        adjacent_positions = ((up == value) | (down == value) | (left == value) | (right == value))# | (upper_left == value) | (upper_right == value) | (lower_left == value) | (lower_right == value))
        # print(value, adjacent_positions.device)
        return adjacent_positions
    def linear_scale(self, tensor, min_value, max_value):
        # Calculate the current min and max values of the tensor
        current_min = torch.min(tensor)
        current_max = torch.max(tensor)
        
        # Scale the tensor linearly to the specified range
        scaled_tensor = (tensor - current_min) * (max_value - min_value) / (current_max - current_min) + min_value
        
        return scaled_tensor


class GradientRegularizer(nn.Module):
    def __init__(self, kernel_sizes=[3, 5, 7, 9, 11], mode='AE'):
        """
        Initializes the Weighted MAE Loss.
        
        Args:
            kernel_sizes (list): List of kernel sizes to define the receptive field.
            mode (str): 'AE' for Absolute Error, 'SE' for Squared Error.
        """
        super(GradientRegularizer, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.mode = mode

    def z_scoare_normalize(self, image):
        
        non_zero_pixels = image[image != 0].ravel()
        
        mean = torch.mean(non_zero_pixels)
        std = torch.std(non_zero_pixels)
        print("mean: ", mean, " std: ", std)
        
        # Normalize the image
        normalized_image = (image - mean) / (std + 1e-8)  # Add a small value to avoid division by zero
        
        return normalized_image
    
    def min_max_normalize(self, image, reverse=True):
        """
        Perform min-max normalization on the image, scaling values to [0, 1].
        Args:
        - image: A PyTorch tensor representing the image.
                Shape: (C, H, W) or (B, C, H, W) for 2D images or (B, C, D, H, W) for 3D images.
                
        Returns:
        - normalized_image: The image after min-max normalization, with values in the range [0, 1].
        """
        # Compute min and max values (excluding zero pixels)
        
        non_zero_pixels = image[image != 0].ravel()
        min_val = torch.min(non_zero_pixels)
        max_val = torch.max(non_zero_pixels)

        # print("min_val: ", min_val, " max_val: ", max_val)

        
        # Min-max normalization formula
        normalized_image = (image - min_val) / (max_val - min_val + 1e-5)  # Add small epsilon to avoid division by zero
        if reverse:
            return 1-normalized_image
        else:
            return normalized_image

    def compute_weight(self, tissue_intensity, kernel_size):
        """
        Compute pixel-wise weight based on similarity to neighboring pixels.
        
        Args:
            input1 (torch.Tensor): The input image (batch_size, channels, height, width).
            kernel_size (int): The size of the receptive field.
        
        Returns:
            torch.Tensor: Pixel-wise weight tensor.
        """
        padding = kernel_size // 2
        unfold = F.unfold(tissue_intensity, kernel_size=kernel_size, padding=padding) # (batch_size, channels * kernel_size^2, height * width)
        unfolded_input = unfold.view(tissue_intensity.size(0), tissue_intensity.size(1), kernel_size, kernel_size, tissue_intensity.size(2), tissue_intensity.size(3)) # (batch_size, channels, kernel_size, kernel_size, height, width)
        
        # Compute similarity between center pixel and its neighbors
        center_pixel = tissue_intensity.unsqueeze(2).unsqueeze(3) # (batch_size, channels, 1, 1, height, width)

        weight = torch.abs(center_pixel - unfolded_input).mean(dim=(2, 3))
        # if self.mode == 'AE':
        #     weight = torch.abs(center_pixel - unfolded_input).mean(dim=(2, 3)) # (batch_size, channels, height, width)
        # elif self.mode == 'SE':
        #     weight = torch.square(center_pixel - unfolded_input).mean(dim=(2, 3)) # (batch_size, channels, height, width)
        # else:
        #     raise ValueError("Mode should be 'AE' or 'SE'.")
        # print(weight.shape)

        
        return weight

    
    def plot_histogram(self, image, filename):
        
        # non_zero_pixels = image[image != 0].ravel()
        plt.figure(figsize=(6, 4))
        plt.hist(image.ravel(), bins=50, color='blue', alpha=0.7)
        plt.title('Histogram of Image Intensities')
        plt.xlabel('Intensity Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
        plt.savefig("/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/main/utils/hist_{}.png".format(filename))

    
    def forward(self, input1, tissue_intensity, input2):

        """
        Forward pass of the Weighted MAE loss.

        Args:
            input1 (torch.Tensor): Predicted image (batch_size, channels, height, width).
            input2 (torch.Tensor): Ground truth image (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Computed weighted MAE loss.
        """

        if self.mode == 'AE':
            differences = torch.abs(input1 - input2)
        elif self.mode == 'SE':
            differences = torch.square(input1 - input2)
        elif self.mode == 'Gradient':
            grad_x = torch.abs(input2[:, :, 1:, :] - input2[:, :, :-1, :])  # Gradient along x direction
            grad_x = torch.nn.functional.pad(grad_x, (0, 0, 0, 1))  # Pad height to maintain shape
            grad_y = torch.abs(input2[:, :, :, 1:] - input2[:, :, :, :-1])  # Gradient along y direction
            grad_y = torch.nn.functional.pad(grad_y, (0, 1, 0, 0))


            # print(grad_x.shape, grad_y.shape)
            # Compute gradient magnitude
            differences = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-5)
            # print(differences.shape)

        
        


        total_penalty = torch.zeros_like(differences)
        print(torch.mean(differences))

        for kernel_size in self.kernel_sizes:

            # Calculate the similarity to neighboring pixel
            weight = self.compute_weight(tissue_intensity, kernel_size)

            
            # weight_norm=self.min_max_normalize(weight, reverse=False)

            # mask=torch.where(tissue_intensity<0.2, torch.tensor(0.0), torch.tensor(1.0))
            # weight_norm_mask=weight_norm*mask

            # weight_norm_mask_zscore=self.z_scoare_normalize(weight_norm_mask)

            # weight_norm_mask_zscore_mincutoff=weight_norm_mask_zscore-torch.min(weight_norm_mask_zscore)

            # weight_norm_mask_zscore_mincutoff_maxcutoff=torch.where(weight_norm_mask_zscore_mincutoff<3, weight_norm_mask_zscore_mincutoff, 3)

            # # print(torch.min(weight_norm_mask_zscore_positive), torch.max(weight_norm_mask_zscore_positive))
            # # print(torch.sum(torch.where(weight_norm_mask_zscore_positive<2, torch.tensor(0.0), torch.tensor(1.0))))
            # # weight_norm_mask_zscore_mask=torch.where(weight_norm_mask_zscore<0, torch.tensor(0.0), weight_norm_mask_zscore)


            # plt.figure(figsize=(15, 15))
            # plt.subplot(2,3,1)
            # plt.imshow(tissue_intensity.numpy()[0,0], cmap='gray')
            # plt.axis('off')
            # plt.colorbar()
            # plt.subplot(2,3,2)
            # plt.imshow(weight_norm.numpy()[0,0], cmap='gray')
            # plt.axis('off')
            # plt.colorbar()
            # plt.show()
            # plt.subplot(2,3,3)
            # plt.imshow(weight_norm_mask.numpy()[0,0], cmap='gray')
            # plt.axis('off')
            # plt.colorbar()
            # plt.show()

            # plt.subplot(2,3,4)
            # plt.imshow(weight_norm_mask_zscore.numpy()[0,0], cmap='gray')
            # plt.axis('off')
            # plt.colorbar()
            # plt.subplot(2,3,5)
            # plt.imshow(weight_norm_mask_zscore_mincutoff.numpy()[0,0], cmap='gray')
            # plt.axis('off')
            # plt.colorbar()
            # plt.subplot(2,3,6)
            # plt.imshow(weight_norm_mask_zscore_mincutoff_maxcutoff.numpy()[0,0], cmap='gray')
            # plt.axis('off')
            # plt.colorbar()

            # plt.show()


            # plt.savefig('/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/main/utils/demo.png')


            # tissue_intensity_norm=self.min_max_normalize(torch.abs(self.z_scoare_normalize(tissue_intensity)))

            # First, calculate the alternative to segmented tissue map
            # Then, calculate gradient penalty depending on the tissue type following the descending intensity from CSF, WM and GM
            
            # plt.figure()
            # plt.subplot(2,2,1)
            # plt.imshow(tissue_intensity.numpy()[0,0], cmap='gray')
            # plt.axis('off')
            # plt.colorbar()
            # # self.plot_histogram(tissue_intensity, "1")
            
            # mask=torch.where(tissue_intensity<0.2, torch.tensor(0.0), torch.tensor(1.0))
            # tissue_intensity=tissue_intensity*mask
            # # tissue_intensity=self.min_max_normalize(tissue_intensity, reverse=False)
            # plt.subplot(2,2,2)
            # plt.imshow(mask.numpy()[0,0], cmap='gray')
            # plt.axis('off')
            # plt.colorbar()
            # # self.plot_histogram(tissue_intensity, "2")

        
            # tissue_intensity=self.min_max_normalize(tissue_intensity)
            # plt.subplot(2,2,3)
            # plt.imshow(tissue_intensity.numpy()[0,0], cmap='gray')
            # plt.axis('off')
            # plt.colorbar()
            # # self.plot_histogram(tissue_intensity, "3")
            
            # tissue_intensity=tissue_intensity*mask
            # plt.subplot(2,2,4)
            # plt.imshow(tissue_intensity.numpy()[0,0], cmap='gray')
            # plt.axis('off')
            # plt.colorbar()
            # # self.plot_histogram(tissue_intensity, "4")
            
            # plt.show()
            # plt.savefig('/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/main/utils/{}_{}.png'.format(str(kernel_size),self.mode))
            
            
            


            gradient_penalty=tissue_intensity - weight

            # tissue_intensity_norm_weight = self.compute_weight(tissue_intensity_norm, kernel_size)

            total_penalty += gradient_penalty
            
            # plt.figure()
            
            # plt.subplot(2,2,1)
            # plt.imshow(tissue_intensity.numpy()[0,0], cmap='gray')
            # plt.axis('off')
            # plt.colorbar()
            
            # plt.subplot(2,2,2)
            # anatomical_input = self.z_scoare_normalize(tissue_intensity)
            # plt.imshow(anatomical_input.numpy()[0,0], cmap='gray')
            # plt.axis('off')
            # plt.colorbar()
            
            # plt.subplot(2,2,3)
            # anatomical_input = torch.abs(anatomical_input)
            # # anatomical_input = torch.where(anatomical_input < 0, torch.tensor(0.0), anatomical_input)
            # plt.imshow(anatomical_input.numpy()[0,0], cmap='gray')
            # plt.axis('off')
            # plt.colorbar()
            
            # plt.subplot(2,2,4)
            # # anatomical_input=1/(anatomical_input+ 1e-8)
            # anatomical_input=self.min_max_normalize(anatomical_input)

            # plt.imshow(anatomical_input.numpy()[0,0], cmap='gray')
            # plt.axis('off')
            # plt.colorbar()
        
        

            # plt.figure()
            # plt.subplot(2,3,1)
            # plt.imshow(input1.numpy()[0,0], cmap='jet')
            # plt.axis('off')
            # plt.colorbar()

            # plt.subplot(2,3,2)
            # plt.imshow(tissue_intensity.numpy()[0,0], cmap='gray')
            # plt.axis('off')
            # plt.colorbar()

            # plt.subplot(2,3,3)
            # plt.imshow(input2.numpy()[0,0], cmap='jet')
            # plt.axis('off')
            # plt.colorbar()

            # plt.subplot(2,3,4)
            # plt.title('Differences')
            # plt.imshow(differences.numpy()[0,0], cmap='jet')
            # plt.axis('off')
            # plt.colorbar()

            # plt.subplot(2,3,5)
            # plt.title('Penalty')
            # plt.imshow(gradient_penalty.numpy()[0,0], cmap='jet')
            # plt.axis('off')
            # plt.colorbar()

            # plt.subplot(2,3,6)
            # plt.title('Loss')
            # plt.imshow(gradient_penalty.numpy()[0,0]*differences.numpy()[0,0], cmap='jet')
            # plt.axis('off')
            
            # plt.colorbar()

            # plt.show()
            # plt.savefig('/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/main/utils/{}_{}.png'.format(str(kernel_size),self.mode))



        total_penalty /= len(self.kernel_sizes)  # Normalize by the number of kernel sizes
        regularized_differences = differences * total_penalty  # Apply total_penalty to the differenes
        loss = regularized_differences.mean()  # Take the mean over all pixels

        return loss
    
    
if __name__ == "__main__":
    mae_loss = GradientRegularizer(kernel_sizes=[3])
    mse_loss = GradientRegularizer(kernel_sizes=[3], mode='SE')
    grad_loss = GradientRegularizer(kernel_sizes=[3], mode='Gradient')


    # mse_loss = GradientRegularizer(mode='SE')

    PET_input=np.load('/home/linyunong/project/src/AD_FDG_250/002_S_1268.nii/002_S_1268_Axial_40.npy')[0]
    MRI_full=np.load('/home/linyunong/project/src/AD_MRI_250/002_S_1268.nii/002_S_1268_Axial_40.npy')[0]
    MRI_input=np.load('/home/linyunong/project/src/AD_MRI_250/002_S_1268.nii/002_S_1268_Axial_40.npy')[3]
    PET_pred=np.array(Image.open("/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_images/ConfigC/fold0/2024-9-11-94614/fusion_output/AD/002_S_1268.nii/002_S_1268_Axial_40.tif"))
    # PET_pred=PET_input+torch.rand(MRI_input.shape)/10

    PET_input=torch.from_numpy(PET_input).unsqueeze(0).unsqueeze(1)
    MRI_full=torch.from_numpy(MRI_full).unsqueeze(0).unsqueeze(1)
    MRI_input=torch.from_numpy(MRI_input).unsqueeze(0).unsqueeze(1)
    PET_pred=torch.from_numpy(PET_pred).unsqueeze(0).unsqueeze(1)

    print(MRI_input.shape)
    print(PET_pred.shape)


    print(PET_input.shape, PET_pred.shape, MRI_full.shape, MRI_input.shape)

    # MRI_input = torch.tensor([[[[1.0, 1.0, 0.25],
    #                         [1.0, 0.0, 0.25],
    #                         [0.25, 0.0, 0.25]]]] * 2, dtype=torch.float32)
    # PET_pred = torch.tensor([[[[0.5, 0.2, 0.1],
    #                         [0.2, 0.3, 0.2],
    #                         [0.1, 0.9, 0.0]]]] * 2, dtype=torch.float32)
    
    # PET_input = torch.tensor([[[[0.7, 0.5, 0.3],
    #                         [0.6, 0.7, 0.4],
    #                         [0.3, 1.0, 0.2]]]] * 2, dtype=torch.float32)
    # print(PET_input.shape, PET_pred.shape, MRI_input.shape)
    PET_mae_loss=mae_loss(PET_input, MRI_full, PET_pred)
    PET_mse_loss=mse_loss(PET_input, MRI_full, PET_pred)
    PET_grad_loss=grad_loss(PET_input, MRI_full, PET_pred)


    # MRI_mae_loss=mae_loss(MRI_input, PET_pred, MRI_input)
    # PET_mse_loss=mse_loss(PET_input, PET_pred, MRI_input)
    # MRI_mse_loss=mse_loss(MRI_input, PET_pred, MRI_input)

    print("MAE Loss:", PET_mae_loss.item(), PET_mse_loss.item(), PET_grad_loss.item())
    # print("MSE Loss:", PET_mse_loss.item(), MRI_mse_loss.item())