import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.ndimage import uniform_filter1d

def preprocess_gyro_data(train_seq):
    processed_df = train_seq.copy()
    rot_cols = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
    
    for idx, row in processed_df.iterrows():
        for col in rot_cols:
            signal = np.array(row[col])
            
            if np.any(np.isnan(signal)):
                signal = handle_nan_values(signal)
            
            signal_detrended = remove_drift_safe(signal)
            signal_filtered = apply_highpass_filter(signal_detrended, cutoff=0.5, fs=50)
            signal_enhanced = adaptive_amplification(signal_filtered)
            signal_denoised = smart_denoising(signal_enhanced)
            
            processed_df.at[idx, col] = signal_denoised.tolist()
    
    return processed_df

def handle_nan_values(signal):
    if np.all(np.isnan(signal)):
        return np.zeros_like(signal)
    
    mask = ~np.isnan(signal)
    if np.sum(mask) > 1:
        indices = np.arange(len(signal))
        signal_clean = np.interp(indices, indices[mask], signal[mask])
    else:
        mean_val = np.nanmean(signal)
        if np.isnan(mean_val):
            mean_val = 0.0
        signal_clean = np.full_like(signal, mean_val)
    
    return signal_clean

def remove_drift_safe(signal):
    if np.any(~np.isfinite(signal)):
        signal = handle_nan_values(signal)
    
    if np.ptp(signal) < 1e-12:
        return signal - np.mean(signal)
    
    x = np.arange(len(signal))
    
    finite_mask = np.isfinite(signal) & np.isfinite(x)
    
    if np.sum(finite_mask) < 3:
        if np.sum(finite_mask) >= 2:
            coeffs = np.polyfit(x[finite_mask], signal[finite_mask], deg=1)
            trend = np.polyval(coeffs, x)
        else:
            trend = np.full_like(signal, np.mean(signal[finite_mask]) if np.sum(finite_mask) > 0 else 0)
    else:
        coeffs = np.polyfit(x[finite_mask], signal[finite_mask], deg=2)
        trend = np.polyval(coeffs, x)
    
    return signal - trend

def apply_highpass_filter(signal, cutoff=0.5, fs=50, order=4):
    if np.ptp(signal) < 1e-12:
        return signal
    
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist
    
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.95
    
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)

def adaptive_amplification(signal, percentile_threshold=75):
    if np.ptp(signal) < 1e-12:
        return signal
    
    window_size = min(50, max(3, len(signal) // 10))
    local_energy = uniform_filter1d(signal**2, size=window_size)
    
    threshold = np.percentile(local_energy, percentile_threshold)
    gain_factor = np.where(local_energy > threshold, 1.0, 3.0)
    
    if len(gain_factor) >= 5:
        window_length = min(21, len(gain_factor))
        if window_length % 2 == 0:
            window_length -= 1
        gain_factor = savgol_filter(gain_factor, 
                                   window_length=window_length, 
                                   polyorder=min(3, window_length-1))
    
    return signal * gain_factor

def smart_denoising(signal, window_length=11, polyorder=3):
    if len(signal) < 5:
        return signal
    
    window_length = min(window_length, len(signal))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 3:
        window_length = 3
    
    polyorder = min(polyorder, window_length - 1)
    
    smoothed = savgol_filter(signal, window_length, polyorder)
    
    signal_std = np.std(signal)
    if signal_std > 1e-12:
        significant_events = np.abs(signal) > 2 * signal_std
        result = smoothed.copy()
        result[significant_events] = signal[significant_events]
        return result
    else:
        return smoothed
            
def integrate_angular_velocity(train_seq):
    processed_df = train_seq.copy()
    rot_cols = ['rot_x', 'rot_y', 'rot_z']
    dt = 1/50
    
    for idx, row in processed_df.iterrows():
        for col in rot_cols:
            angular_velocity = np.array(row[col])
            
            if np.any(np.isnan(angular_velocity)):
                angular_velocity = handle_nan_values(angular_velocity)
            
            angles = np.cumsum(angular_velocity) * dt
            angles_detrended = remove_drift_safe(angles)
            
            processed_df.at[idx, col] = angles_detrended.tolist()
    
    return processed_df