import numpy as np

def normalized(x):
    shape = x.shape
    if len(shape) == 3:
        nx = np.zeros(shape)
        for i in range(shape[0]):
            nx[i, :, :] = (x[i] - np.min(x[i]))/(np.max(x[i]) - np.min(x[i]))
    else:
        nx = (x - np.min(x)) / (np.max(x) - np.min(x))
    return nx

def mse(x, y):
    return np.means(np.abs(x - y) ** 2)

def psnr(x, y):
    assert x.shape == y.shape
    assert x.dtype == y.dtype or np.issubdtype(x.dtype, np.float32) \
           and np.issubdtype(y.dtype, np.float32)
    if x.dtype == np.uint8:
        max_intensity = 256
    else:
        max_intensity = 1

    mse = np.sum((x - y) ** 2).astype(float) / x.size
    return 20 * np.log10(max_intensity) - 10 * np.log10(mse)