import torch
import numpy as np


def real2complex_torch(x):
    y = x[:, 0, :, :] + 1j * x[:, 1, :, :]
    return y

def complex2real_b_troch(x):
    size = x.shape
    y = torch.zeros((size[0], 2, size[1], size[2]), dtype=torch.float32).to(x.device)
    y[:, 0, :, :] = x.real
    y[:, 1, :, :] = x.imag
    return y

def fft2(x, norm='ortho'):
    return torch.fft.fft2(x, norm=norm)

def ifft2(x, norm='ortho'):
    return torch.fft.ifft2(x, norm=norm)


def Gaussian_selection(small_acs_block, input_data, input_mask, std_scale=4):
    nrow, ncol = input_data.shape[1], input_data.shape[2]
    center_kx = int(nrow / 2)
    center_ky = int(ncol / 2)

    temp_mask = np.copy(input_mask)
    temp_mask[center_kx - small_acs_block[0] // 2:center_kx + small_acs_block[0] // 2,
    center_ky - small_acs_block[1] // 2:center_ky + small_acs_block[1] // 2] = 0

    loss_mask = np.zeros_like(input_mask)
    count = 0

    while count <= int(np.ceil(np.sum(input_mask[:]) * rho)):

        indx = int(np.round(np.random.normal(loc=center_kx, scale=(nrow - 1) / std_scale)))
        indy = int(np.round(np.random.normal(loc=center_ky, scale=(ncol - 1) / std_scale)))

        if (0 <= indx < nrow and 0 <= indy < ncol and temp_mask[indx, indy] == 1 and loss_mask[indx, indy] != 1):
            loss_mask[indx, indy] = 1
            count = count + 1
    trn_mask = input_mask - loss_mask

    return trn_mask, loss_mask

def uniform_selection(small_acs_block, rho, input_data, input_mask):

    nrow, ncol = input_data.shape[1], input_data.shape[2]

    center_kx = int(find_center_ind(input_data, axes=(0, 2)))
    center_ky = int(find_center_ind(input_data, axes=(0, 2)))

    temp_mask = np.copy(input_mask)
    temp_mask[center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
    center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

    pr = np.ndarray.flatten(temp_mask)
    ind = np.random.choice(np.arange(nrow * ncol),
                           size=int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))

    [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))

    loss_mask = np.zeros_like(input_mask)
    loss_mask[ind_x, ind_y] = 1

    trn_mask = input_mask - loss_mask

    return trn_mask, loss_mask

def find_center_ind(kspace, axes=(1, 2, 3)):
    center_locs = norm(kspace, axes=axes).squeeze()
    return np.argsort(center_locs)[-1:]

def index_flatten2nd(ind, shape):
    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))
    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]