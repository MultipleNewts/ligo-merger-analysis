# %%
import numpy as np
import matplotlib.pyplot as plt
from analysis_techniques.ligonoise import LIGOEvent
from analysis_techniques.welch_method import welch
from scipy.signal.windows import blackman, tukey
from scipy.signal import butter, filtfilt


# %%
def whiten(data, fs, noise_freqs, noise_psd, window_func=tukey):
    """
    Whitens data by factoring out a given frequency profile.

    Parameters
    ----------
    data : `array`
        the signal data to whiten
    fs : `int`
        the sampling frequency of the data sample
    noise_freqs : `array`
        the frequency domain of the PSD
    noise_psd : `array`
        the power spectral density to be factored out
    window_func : `FunctionType`
        function pointer for window function, defaults to ######

    Returns
    -------
    white_data : `array`
        the whitened data signal in the time domain
    """
    # defines useful variables
    data_length = data.shape[0]
    w = window_func(data_length)

    # detrends (by constant) and windows adata
    windowed_data = w*(data - np.mean(data))
    # computes power correction for data
    power_correction = np.sum(np.power(w, 2)) / w.shape[0]  # is this the same as mine???

    # Fourier transform data and find frequency domain
    FT_data = np.fft.rfft(windowed_data)
    data_freqs = np.fft.rfftfreq(data_length, 1/fs)

    # whiten data
    interp_PSD = np.interp(data_freqs, noise_freqs, noise_psd)
    whitened_data = FT_data * np.sqrt(2 / power_correction / fs) / np.sqrt(interp_PSD)
    processed_data = np.fft.irfft(whitened_data)
    return processed_data

