# %%
import numpy as np
from scipy.signal.windows import blackman


# %%
# an implementation of Welch's method for finding an averaged PSD
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
    # converts overlap ratio into an integer number of datapoints
    noverlap = int((seglength*overlap)//1)
    # computes the step size for each window movement
    step = seglength - noverlap
    # creates the window function for the required segment length
    w = window_func(seglength)
    # computes constant for normalisation later
    WSum = np.sum(w*w)

    # iterates through each segment and pre-treats data
    segments = []
    max_iter = int((len(data) - seglength)//step) + 1
    for i in range(max_iter):
        index = int(i*step)
        # obtains segment data
        temp_data = data[index:(index+seglength)]
        # detrends data using a "constant" detrend
        detrend_data = temp_data - np.mean(temp_data)
        # applies window function and appends data
        segments.append(w*detrend_data)
    # computes PSD for each segment
    FFT_segments = []
    for segment in segments:
        # computes PSD (i.e. |FFT|^2) and normalises data
        temp = (1/(WSum*fs))*(np.abs(np.fft.rfft(segment))**2)
        FFT_segments.append(temp)
    # finds the averaged PSD
    averaged_PSD = np.mean(FFT_segments, axis=0)
    # accounts for sampling power errors (DC and Nyquist effects)
    averaged_PSD[1:-1] *= 2
    # obtains domain of frequencies
    freqs = np.fft.rfftfreq(seglength, 1/fs)
    return freqs, averaged_PSD
