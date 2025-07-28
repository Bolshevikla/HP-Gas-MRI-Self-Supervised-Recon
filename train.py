import argparse
import pathlib
import time
import torch.nn.functional as F

import numpy as np
import torch
import math
import shutil
import logging
from torch.utils.data import DataLoader
import utils.compressed_sensing as cs
import matplotlib.pyplot as plt
from utils.dataset_Xe import *
from model.model_noiser2noise_sample_Unet_VAE import *
from model.model import *
from tqdm import tqdm


class DataTransformer():
    def __call__(self, ks):
        kspace = torch.from_numpy(np.fft.fftshift(ks))
        return kspace


def create_dataset(args):
    train_data = SliceData(
        root=,
        transform=DataTransformer(),
        batch_size=args.batch_size
    )
    dev_data = SliceData(
        root=,
        transform=DataTransformer(),
        batch_size=args.batch_size
    )
    return train_data, dev_data


def create_data_loader(args):
    train_data, dev_data = create_dataset(args)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, dev_loader

def apply_mask(args, input, shape, acc):
    mask = cs.cartesian_mask(shape, acc=acc, centred=True, sample_n=8)
    mask = torch.from_numpy(mask).to(args.device)
    under_k = mask * input
    return under_k, mask

def get_reciprocal(mask):
    shape = mask.shape
    weight = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if mask[i, j] != 0:
                weight[i, j] = 1 / mask[i, j]
    return weight


def Log_UP(K_min, K_max, epoch):
    Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
    return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / args.epochs * epoch)]).float().to(args.device)


def kl_divergence(mu_layers, logvar_layers):
    gmm_loss = logvar_layers
    return gmm_loss

def check_for_nan_inf(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN or Inf found in {name}")

def add_noise(input, sigm):
    shape = input.shape
    noise = (np.random.normal(0, sigm, shape).astype(np.float32)) + 1j * (np.random.normal(0, sigm, shape).astype(np.float32))
    noise = torch.from_numpy(noise).to(input.device)
    x = input + noise
    return x

def train_epoch(args, epoch, model, epoch_iterator, optimizer, writer, criterion1, criterion2,T_max, T_min, P_mask):
    model.train()

    t = Log_UP(T_min, T_max, epoch)
    if t < 1:
        k = 1 / t
    else:
        k = torch.tensor([1]).float().to(args.device)

    model.k = k
    model.t = t

    losses = []
    kl_losses = []
    recon_losses = []
    reg_losses = []
    global_step = epoch * len(epoch_iterator)
    
    for iter, data in enumerate(epoch_iterator):
        kspace = data.to(args.device)

        max = []
        gt_img = torch.fft.ifft2(kspace)

        gt_img = F.pad(gt_img, (6, 6, 0, 0), mode='constant', value=0)

        for i in range(gt_img.shape[0]):
            gt_img[i, :, :] = gt_img[i, :, :] / torch.max(torch.abs(gt_img[i, :, :]))
        kspace = torch.fft.fft2(gt_img)

        img_recon, mu, sigma, reconk, subk, mask_omega, mask_lambda, us_img, doubleus_img, recon_k, reg_loss = model(kspace)

        #loss
        kl_loss = kl_divergence(mu, sigma)
        weight_loss = criterion2(reconk, subk) + criterion1(reconk, subk) + (1e-4)*kl_loss + (1e-4)*reg_loss

        optimizer.zero_grad()
        weight_loss.backward()
        optimizer.step()

        losses.append(weight_loss.item())
        kl_losses.append(kl_loss.item())
        recon_losses.append(criterion2(recon_k, complex2real_b_troch(kspace)).item())
        reg_losses.append(reg_loss.item())
    return np.mean(losses), np.mean(kl_losses), np.mean(reg_losses), np.mean(recon_losses)


def evaluate(args, epoch, model, epoch_iterator, writer, criterion1, criterion2):
    model.eval()
    losses = []
    kl_losses = []
    recon_losses = []
    reg_losses = []
    global_step = epoch * len(epoch_iterator)
    for iter, data in enumerate(epoch_iterator):
        kspace = data.to(args.device)
        gt_img = torch.fft.ifft2(kspace)

        gt_img = F.pad(gt_img, (6, 6, 0, 0), mode='constant', value=0)

        for i in range(gt_img.shape[0]):
            gt_img[i, :, :] = gt_img[i, :, :] / torch.max(torch.abs(gt_img[i, :, :]))
        kspace = torch.fft.fft2(gt_img)


        img_recon, mu, sigma, reconk, subk, mask_omega, mask_lambda, us_img, doubleus_img, recon_k, reg_loss = model(kspace)

        kl_loss = kl_divergence(mu, sigma)
        weight_loss = criterion1(reconk, subk) + criterion2(reconk, subk) +  kl_loss + (1e-4)*reg_loss

        losses.append(weight_loss.item())
        recon_losses.append(criterion2(recon_k, complex2real_b_troch(kspace)).item())
    return np.mean(losses), np.mean(recon_losses)


def save_model(args, exp_dir, epoch, model, optimizer, best_eva_loss, is_new_best):
    torch.save(
        {
            'epoch':epoch,
            'args':args,
            'model':model.state_dict(),
            'best_eva_loss':best_eva_loss,
            'optimizer':optimizer.state_dict(),
            'exp_dir':exp_dir,
        },
        f = exp_dir/'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):
    model = Unknown(layer_no=1, desired_sparsity=args.sparsity1,  flag_1D=False, mask_shape=[96, 96], acc=4, device=args.device).to(args.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params/1e6}M')
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    model.load_state_dict(checkpoint['model'])
    optimizer, criterion1, criterion2 = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer, criterion1, criterion2



def build_optim(args, params):
    optim = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.99))
    criterion1 = torch.nn.MSELoss()
    criterion2 = torch.nn.L1Loss()
    return optim, criterion1, criterion2


def main(args):
    args.epx_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.resume:
        checkpoint, model, optimizer, criterion1, criterion2 = load_model(args.checkpoint)
        best_eva_loss = checkpoint['best_eva_loss']
        start_epoch = 0
        del checkpoint
    else:
        model = build_model(args)
        optimizer, criterion1, criterion2 = build_optim(args, model.parameters())
        best_eva_loss = 1e9
        start_epoch = 0

    logging.info(args)
    logging.info(model)

    train_loader, dev_loader = create_data_loader(args)
    T_min, T_max = args.t_min, args.t_max

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.8)

    for epoch in range(start_epoch, args.epochs):
        train_iterator = tqdm(train_loader, desc='Train')
        train_loss, kl_loss, reg_loss, recon_loss = train_epoch(args, epoch, model, train_iterator, optimizer, writer, criterion1, criterion2, T_max, T_min)
        scheduler.step()
        dev_iterator = tqdm(dev_loader, desc='Dev')
        eva_loss, eva_recon_loss = evaluate(args, epoch, model, dev_iterator, writer, criterion1, criterion2)
        is_new_best = eva_loss < best_eva_loss
        best_eva_loss = min(best_eva_loss, eva_loss)
        save_model(args, args.epx_dir, epoch, model, optimizer, best_eva_loss, is_new_best)

    writer.close()

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate')
    parser.add_argument('--sparsity1', type=float, default=0.6, help='The first mask')
    parser.add_argument('--report_interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--t_max', type=int, default=10, help='')
    parser.add_argument('--t_min', type=float, default=0.1, help='')

    parser.add_argument('--epx_dir', type=pathlib.Path, default=r'',
                        help='Path where model and result should be saved')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Which device to train on. Set to "cuda:n" ,n represent the GPU number')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=False,
                        default=r"",
                        help='Path to the model')
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    main(args)


