import numpy as np
from configs.config import cfg

def computeWaveletSize(fc, nc, fs):
    sd = (nc / 2) * (1 / np.abs(fc)) / cfg.morlet_sd_factor
    return int(2 * np.floor(np.round(sd * fs * cfg.morlet_sd_spread) / 2) + 1)

def gausswin(size, alpha):
    halfSize = int(np.floor(size / 2))
    idiv = alpha / halfSize
    t = (np.arange(size, dtype=np.float64) - halfSize) * idiv
    window = np.exp(-(t * t) * 0.5)
    return window

def morlet(fc, nc, fs):
    size = computeWaveletSize(fc, nc, fs)
    half = int(np.floor(size / 2))
    gauss = gausswin(size, cfg.morlet_sd_spread / 2)
    igsum = 1 / gauss.sum()
    ifs = 1 / fs
    
    t = (np.arange(size, dtype=np.float64) - half) * ifs
    wavelet = gauss * np.exp(2 * np.pi * fc * t * 1j) * igsum
    return wavelet

def fractional(x):
    return x - int(x)