import argparse
import shutil
import time
import pathlib
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils import my_math
from utils.dataset_Xe import *
from utils.my_math import *
from torch.utils.data import DataLoader
from utils import compressed_sensing as cs
from skimage.metrics import structural_similarity as ssim
from tensorboardX import SummaryWriter
from utils import metrics
from tqdm import tqdm
from model.model_noiser2noise_VAE import *

class DataTransformer():
    def __call__(self, ks):
        kspace = torch.from_numpy(np.fft.fftshift(ks))
        return kspace


def create_data(args):
    test_data = SliceData(
        root=,
        transform=DataTransformer(),
        batch_size=args.batch_size
    )
    return test_data

def create_data_loader(args):
    test_data = create_data(args)
    dev_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return dev_loader

def add_noise(input, sigm):
    shape = input.shape
    noise = (np.random.normal(0, sigm, shape).astype(np.float32)) + 1j * (np.random.normal(0, sigm, shape).astype(np.float32))
    noise = torch.from_numpy(noise).to(input.device)
    x = input + noise
    return x

def test_recon(args, model, data_loader):
    st = time.perf_counter()
    losses = []
    num = 0
    PSNR_zf = []
    SSIM_zf = []
    PSNR_recon = []
    SSIM_recon = []
    mask_theta_list = []
    mask_lambda_list = []
    for iter, data in enumerate(data_loader):
        kspace = data.to(args.device)

        gt_img = torch.fft.ifft2(kspace)

        gt_img = F.pad(gt_img, (6, 6, 0, 0), mode='constant', value=0)

        for i in range(gt_img.shape[0]):
            gt_img[i, :, :] = gt_img[i, :, :] / torch.max(torch.abs(gt_img[i, :, :]))
        kspace = torch.fft.fft2(gt_img)

        img_recon, mu, sigma, reconk, subk, mask_omega, mask_lambda, us_img, doubleus_img, recon_k, reg_loss = model(kspace, flag='test')

        img_recon = real2complex_torch(img_recon).detach().cpu().numpy()
        gt_img = (gt_img).cpu().numpy()
        us_img = real2complex_torch(us_img).cpu().numpy()

        #计算指标
        norm_img_recon = metrics.normalized(np.abs(img_recon))
        norm_gt_img = metrics.normalized(np.abs(gt_img))
        norm_us_img = metrics.normalized(np.abs(us_img))

        shape = img_recon.shape
        for i in range(shape[0]):
            psnr_recon = metrics.psnr(norm_img_recon[i], norm_gt_img[i])
            psnr_zf = metrics.psnr(norm_us_img[i], norm_gt_img[i])
            PSNR_recon.append(psnr_recon)
            PSNR_zf.append(psnr_zf)

            ssim_recon = ssim(norm_img_recon[i], norm_gt_img[i], data_range=norm_img_recon.max() - norm_img_recon.min())
            ssim_zf = ssim(norm_us_img[i], norm_gt_img[i], data_range=norm_us_img.max() - norm_us_img.min())
            SSIM_recon.append(ssim_recon)
            SSIM_zf.append(ssim_zf)

            mask_theta_list.append((mask_omega-mask_lambda).detach().cpu().numpy())
            mask_lambda_list.append(mask_lambda.detach().cpu().numpy())

    return np.mean(PSNR_recon), np.mean(SSIM_recon), np.mean(PSNR_zf), np.mean(SSIM_zf), mask_theta_list, mask_lambda_list


def build_model(args):
    model = Unknown(layer_no=5, desired_sparsity=args.sparsity1, flag_1D=False, mask_shape=[96, 96], acc=4, device=args.device).to(args.device)
    num_params = sum(p.numel() for p in model.parameters())
    num_params = num_params / 1e6
    print(f"Number of parameters: {num_params:.2f}M")
    return model

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    weight = 0
    model = build_model(args)
    model.load_state_dict(checkpoint['model'])
    return model, weight

def main(args):
    model, weight = load_model(args.checkpoint)
    data_loader = create_data_loader(args)

    PSNR_recon, SSIM_recon, PSNR_zf, SSIM_zf, mask_theta_list, mask_lambda_list = test_recon(args, model, data_loader)

    print('recon_psnr=', PSNR_recon, 'recon_ssim', SSIM_recon, 'zf_psnr', PSNR_zf, 'zf_ssim', SSIM_zf)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int, help='Mini batch size')
    parser.add_argument('--sparsity1', type=float, default=0.4, help='The first mask')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Which device to train on. Set to "cuda:n" ,n represent the GPU number')

    parser.add_argument('--checkpoint', type=pathlib.Path, required=False,
                        default=r'',
                        help='Path to the model')
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    main(args)
