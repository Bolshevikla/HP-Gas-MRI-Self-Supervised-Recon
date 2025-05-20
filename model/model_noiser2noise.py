import numpy as np
import torch
import torch.nn as nn
import utils.compressed_sensing as cs
from utils.my_math import *
import torch.nn.functional as F
from model.Xe_Ladder import *

dtype = torch.cuda.FloatTensor

def convblock(n_ch, nd, nf=64, ks=3, pad=1, n_out=None):
    if n_out is None:
        n_out = n_ch

    conv = nn.Conv2d

    def conv_i():
        return conv(nf, nf, ks, stride=1, padding=pad, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad, bias=True)

    relu = nn.LeakyReLU()

    layers = [conv_1, relu]
    for i in range(nd-2):
        layers += [conv_i(), relu]

    layers += [conv_n]
    return nn.Sequential(*layers)


class residualblock(nn.Module):
    def __init__(self, n_ch, nf=64, ks=3, pad=1, n_out=None, res_scale=1):
        super().__init__()
        if n_out == None:
            self.n_out = n_ch
        self.nf = nf
        self.ks = ks
        self.pad = pad
        self.conv1 = nn.Conv2d(n_ch, nf, kernel_size=ks, padding=pad, stride=1)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=ks, padding=pad, stride=1)
        self.activation = nn.ReLU(inplace=True)
        self.scale = res_scale

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        out = x * self.scale + input
        return out

def data_consistency(orig_k, input, mask):
    ks = torch.fft.fft2(real2complex_torch(input))
    k_dc = ((1 - mask) * ks + orig_k)
    img_dc = complex2real_b_troch(torch.fft.ifft2((k_dc)))
    return img_dc


class ResNet(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.nc = nc
        self.resconv = residualblock(n_ch=64)
        resblocks = []

        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1, stride=1)

        self.convn = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)

        self.convnn = nn.Conv2d(64, 2, kernel_size=3, padding=1, stride=1)

        for i in range(self.nc):
            resblocks.append(residualblock(n_ch=64))
        self.res = nn.ModuleList(resblocks)
    def forward(self, input):
        """
        mask为mask_lambda，第二个欠采样mask
        """
        x = input
        x = self.conv1(x)
        inter = x
        for i in range(self.nc):
            x = self.res[i](x)
        x = self.convn(x)
        x = x + inter
        x = self.convnn(x)
        return x

class BinaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = input.new(input.size())
        out[input >= 0] = 1
        out[input < 0] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t * 2), 2)) * grad_output
        return grad_input, None, None, None

class Unknown(nn.Module):
    def __init__(self, layer_no, desired_sparsity,  flag_1D, mask_shape=[96, 96], acc=4, device='cuda:0'):
        super().__init__()
        self.mask_shape = mask_shape if not flag_1D else mask_shape[-1]
        self.desired_sparsity = desired_sparsity
        self.acc = acc
        self.flag_1D = flag_1D
        self.pmask_slope = 5
        self.LayerNo = layer_no

        self.Phi = nn.Parameter(self.initialize_p()).to(device)

        self.MyBinarize = BinaryQuantize.apply

        self.k = torch.tensor([10]).float().to(device)
        self.t = torch.tensor([0.1]).float().to(device)

        self.device = device

        self.weight = self.generate_center_weight(mask_shape).to(device)

        self.model = UNetModel_Xe(in_chans=2, out_chans=2, chans=64, num_pool_layers=4, drop_prob=0., latent_dim=128, components=4)

    def forward(self, full_k, flag='train'):
        create_mask = np.rot90(cs.perturbed_shear_grid_mask(shape=(1, full_k.shape[-2], full_k.shape[-1]), acceleration_rate=self.acc, centred=True, sample_n=8).squeeze())
        mask1 = torch.from_numpy((create_mask).copy()).to(self.device)
        maskp01 = torch.sigmoid(self.pmask_slope * self.Phi)*mask1
        maskpbar1 = maskp01.sum()/(mask1.sum())
        r1 = self.desired_sparsity/maskpbar1
        beta1 = (1 - self.desired_sparsity) / (1 - maskpbar1)
        le1 = torch.le(r1, 1).float()
        maskp1 = le1 * maskp01 * r1 + (1 - le1) * (1 - (1 - maskp01) * beta1)* mask1
        u1 = torch.from_numpy(np.random.uniform(low=0.0, high=1.0, size=maskp01.size())).to(self.device).type(dtype) * mask1

        mask_matrix1 = self.model(maskp1 - u1, self.k, self.t)
        mask2 = mask1 * mask_matrix1
        mask_lambda = mask1 - mask2

        temp = mask_lambda
        mask_lambda = mask2
        mask2 = temp

        sub_k = mask1 * full_k
        doublesub_k = mask2 * sub_k

        doubleus_img = complex2real_b_troch(torch.fft.ifft2(doublesub_k))
        us_img = complex2real_b_troch(torch.fft.ifft2(sub_k))

        if flag == 'train':
            x = doubleus_img
        elif flag == 'test':
            x = us_img

        recon_img, mu_layers, logvar_layers = self.VAE(x)

        if flag == 'train':
            recon_img = data_consistency(doublesub_k, recon_img, mask2)
        elif flag == 'test':
            recon_img = data_consistency(sub_k, recon_img, mask1)
        mu = recon_img

        recon_k = torch.fft.fft2(real2complex_torch(mu))
        weight_reconk = complex2real_b_troch(recon_k*mask_lambda)
        weight_subk = complex2real_b_troch(sub_k*mask_lambda)

        reg_loss = torch.sum((1-self.weight*maskp1))

        return recon_img, mu_layers, logvar_layers, weight_reconk, weight_subk, mask1, mask2, us_img, doubleus_img, complex2real_b_troch(recon_k), reg_loss

    def initialize_p(self, eps=1e-3):
        x = torch.from_numpy(np.random.uniform(low=eps, high=1-eps, size=self.mask_shape)).type(dtype)
        return -torch.log(1. / x - 1.)/self.pmask_slope

    def generate_center_weight(self, size, p=1.5, MD=None):
        H, W = size
        if MD is None:
            MD = torch.sqrt(torch.tensor((H / 2) ** 2 + (W / 2) ** 2))
        cx, cy = size[0] // 2, size[1] // 2
        x = torch.arange(size[0])
        y = torch.arange(size[1])
        X, Y = torch.meshgrid(x, y)

        distance = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        W_LPK = (1 - distance / MD) ** p
        W_LPK = torch.clamp(W_LPK, min=0, max=1)

        return W_LPK








