# %%
import numpy as np
from scipy.signal.windows import tukey
from scipy.signal import butter, sosfilt


# %%
def whiten(
        data, fs, noise_freqs,
        noise_psd, window_func=tukey,
        fourier_output=False
        ):
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
        function pointer for window function, defaults to `tukey`

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
    power_correction = np.sum(np.power(w, 2)) / w.shape[0]

    # Fourier transform data and find frequency domain
    FT_data = np.fft.rfft(windowed_data)
    data_freqs = np.fft.rfftfreq(data_length, 1/fs)

    # whiten data
    interp_PSD = np.interp(data_freqs, noise_freqs, noise_psd)
    whitened_data = (
        FT_data * np.sqrt(2 / power_correction / fs)
        / np.sqrt(interp_PSD)
        )
    if fourier_output is not True:
        processed_data = np.fft.irfft(whitened_data)
        return processed_data
    return whitened_data


def bandpass(data, fs, band_min=35, band_max=350, order=8):
    """
    Bandpasses data between a minimum and maximum frequency.

    Parameters
    ----------
    data : `array`
        the signal data to bandpass
    fs : `int`
        the sampling frequency of the data sample
    band_min : `float`
        the minimum frequency of the bandpass, defaults to `35`
    band_max : `float`
        the maximum frequency of the bandpass, defaults to `350`
    order : `int`
        the number of the times the filter is applied, defaults to `8`

    Returns
    -------
    filtered_data : `array`
        the bandpassed data signal in the time domain
    """
    sos = butter(
        order, [band_min, band_max],
        btype="band", fs=fs, output="sos"
        )
    filtered_data = sosfilt(sos, data)
    return filtered_data


def normalise(data, dt, bound_by_axis=True, constant=1):
    '''
    Parameters
    ---------
    data : `array`
        The data to normalise
    dt : `float`
        The time step between each data point
    bound_by_axis : `bool`
        Whether to normalise over the signed area, or absolute area bound by data and the time axis
    constant : `float`
        The desired integral of the normalised data

    Returns
    ------
    normalised_data : `array`
        The input data scaled by a normalisation factor
    '''
    if bound_by_axis:
        to_integrate = np.abs(data)
    else:
        to_integrate = data

    normalisation_factor = np.sum(to_integrate) * dt

    return data / normalisation_factor * constant
