import numpy as np

# This is just a start, this code doesn't exactly do what you think it may


def Qplot(data, fs, hop_length, fmin, kMax, Q, b):
    """
    Creates a constant Q-

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
    deltaFmin = fmin / Q
    ks = np.linspace(0, kMax, kMax, endpoint=False)
    deltaKs = np.power(2, ks/b) * deltaFmin
    fks = Q * deltaKs
    return fks  # Ignore this is to please flake 8


def const_q_column(segment, window_fn, Q):
    segment_length = segment.shape[0]
    ns = np.linspace(0, segment_length, segment_length, endpoint=False)
    window = window_fn(segment_length)
    normalisation = 1 / segment_length
    sum = np.sum(window *
                 segment *
                 np.exp(complex(0, -2 * np.pi * Q * ns / segment_length)))
    return normalisation * sum
