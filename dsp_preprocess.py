import numpy as np 
import matplotlib.pyplot as plt

import argparse
import multiprocessing as mp
import os
import scipy.signal as signal

# ===================== Normalization =====================
def normalization(x, type_norm="mu_law",mu=256):
    # input x : numpy, shape: (window_size, num_channel)
    # output y: numpy, shape: (window_size, num_channel)

    if type_norm == "none":
        y = x
    elif type_norm == "mvc":    #  maximum voluntary contraction (MVC)
        mvc = 0.002479567551179044 # search over all train dataset (current MVC: 4.088872887106668)
        y = x/mvc
    elif type_norm == "standardization":
        mean = np.mean(x,axis=0)
        std = np.std(x,axis=0)
        y = (x-mean)/std
    elif type_norm == "min_max":
        min_val = np.min(x, axis=0)
        max_val = np.max(x, axis=0)
        y = (x - min_val)/(max_val-min_val)
    elif type_norm == "mu_law":
        y = np.sign(x)*(np.log(1+mu*abs(x)))/(np.log(1+mu))
    else:
        raise TypeError(f'{type_norm} is not defined in type_norm')

    return y

# ===================== Filter =====================
def butter_filter(x, type_filter="highpass", cut_low=20, cut_high = 500, fs=2000, order=4):
    nyq = 0.5 * fs
    if type_filter=="bandpass":
        normal_cut_low = [cut_low / nyq, cut_high / nyq]
    else:
        normal_cut_low = cut_low / nyq
    sos = signal.butter(N=order, Wn=normal_cut_low, btype=type_filter, output='sos')
    # y = signal.sosfilt(sos, x)
    y = signal.sosfiltfilt(sos,x,axis=0)
    return y

def filter(x, type_filter="none", cut_low=20, cut_high = 200, fs=2000, order=3):
    if type_filter != 'none':
        type_filter_name, cut_low, cut_high = type_filter.split('_')
    else:
        type_filter_name = "none"

    if type_filter_name == "none":
        y = x
    elif type_filter_name == "HPF":
        y = butter_filter(x, type_filter="highpass", cut_low=float(cut_low), fs=fs, order=order)
    elif type_filter_name == "LPF":
        y = butter_filter(x, type_filter="lowpass", cut_low=float(cut_low), fs=fs, order=order)
    elif type_filter_name == "BPF":
        y = butter_filter(x, type_filter="bandpass", cut_low=float(cut_low), cut_high=float(cut_high), fs=fs, order=order)
    else:
        raise TypeError(f'{type_filter} is not defined in type_filter')
    return y

# ===================== Other Function =====================
def handle_concatenation(existing_array, new_data, axis=0):
    if existing_array is None:
        return new_data
    else:
        return np.concatenate((existing_array, new_data), axis=axis)

def plot_FFT(x, Fs):
    # Perform the FFT
    n = len(x)  # Length of signal
    freq_axis = np.arange(0, (Fs/2), Fs/n) # Create the frequency vector

    # Plot the FFT
    plt.figure(figsize=(12, 4))
    for i in range(x.shape[1]):
        X = np.fft.fft(x[:,i])/n  # FFT normalized by the number of samples
        X = X[range(int(n/2))]  # Remove the symmetric part of the FFT
        plt.plot(freq_axis, abs(X))

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()