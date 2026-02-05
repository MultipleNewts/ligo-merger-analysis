# %%
import numpy as np
from scipy.signal.windows import blackman


# %%
def welch(data, fs, window_func=blackman, seglength=256, overlap=0.5):
    """
    Applies Welch's Method to signal data to get an averaged PSD/periodogram.

    Parameters
    ----------
    data : `array`
        the signal data to apply Welch's method to
    fs : `int`
        the sampling frequency of the data sample
    window_func : `FunctionType`
        function pointer for window function
    seglength : `int`
        the number of datapoints to be included in each segment during Welch's method
    overlap : `float`
        the ratio of overlap each segment has
        floors to get integer datapoints, defaults to `0.5`

    Returns
    -------
    freqs : `array`
        sample frequencies of data
    avgPSD : `array`
        the power spectral density graph averaged using Welch's method
    """
    noverlap = int((seglength*overlap)//1)
    step = seglength - noverlap
    w = window_func(seglength)
    WSum = np.sum(w*w)
    segments = []
    max_iter = int((len(data) - seglength)//step) + 1
    for i in range(max_iter):
        index = int(i*step)
        temp_data = data[index:(index+seglength)]
        detrend_data = temp_data - np.mean(temp_data)
        segments.append(w*detrend_data)
    FFT_segments = []
    for segment in segments:
        temp = (1/(WSum*fs))*(np.abs(np.fft.rfft(segment))**2)
        FFT_segments.append(temp)
    # averaged_PSD = average_dist(FFT_segments)
    averaged_PSD = np.mean(FFT_segments, axis=0)
    averaged_PSD[1:-1] *= 2
    freqs = np.fft.rfftfreq(seglength, 1/fs)
    return freqs, averaged_PSD
