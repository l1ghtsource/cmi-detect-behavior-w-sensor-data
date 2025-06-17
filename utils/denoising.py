import pywt
from scipy.signal import savgol_filter, butter, filtfilt

def wavelet_denoising_2(x, wavelet='db4'):
    coeffs = pywt.wavedec(x, wavelet, mode='per')
    coeffs[len(coeffs) - 1] *= 0
    coeffs[len(coeffs) - 2] *= 0
    result = pywt.waverec(coeffs, wavelet, mode='per')
    if len(x) % 2 == 1:
        result = result[:-1]
    return result

def savgol_denoising(x, window_length=5, polyorder=2):
    if len(x) < window_length:
        return x.copy()
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, len(x))
    if window_length < 3:
        return x.copy()
    polyorder = min(polyorder, window_length - 1)
    return savgol_filter(x, window_length, polyorder)

def butterworth_denoising(x, cutoff_freq=0.1, order=4):
    if len(x) < 6:
        return x.copy()
    nyquist = 0.5
    normal_cutoff = cutoff_freq / nyquist
    normal_cutoff = min(max(normal_cutoff, 0.01), 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, x)
    return filtered_signal

def apply_denoising(data, method):
    if method == 'kalman':
        print('poka net kalmana, soryan')
        return data.copy() # TODO: add kalman
    elif method == 'savgol':
        return savgol_denoising(data)
    elif method == 'butter':
        return butterworth_denoising(data)
    elif method == 'wavelet':
        return wavelet_denoising_2(data)
    else:
        print('wtf is this bro??')
        return data.copy()