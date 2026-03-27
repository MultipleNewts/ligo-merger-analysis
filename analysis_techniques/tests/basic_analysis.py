import numpy as np
from analysis_techniques.welch_method import welch
from gwpy.timeseries import TimeSeries


def get_strain(t0, length, detector="H1", frequency=4096):
    """
    Collects raw data from GWOSC

    Parameters
    ----------
    t0 : 'float'
        The GPS time of the midpoint of data that is being requested
    length : 'float'
        The total duration on data requested in seconds
    detector : 'string'
        Observatory to request data from. "H1" or "L1"
    frequency : 'int'
        Sample rate of data requested

    Returns
    -------
    times : 'array'
        Time stamps of each data point
    values : 'array'
        Strain of each data point
    """
    start_time = t0 - length/2
    end_time = t0 + length/2
    data = TimeSeries.fetch_open_data(detector,
                                      start_time,
                                      end_time,
                                      sample_rate=frequency)
    return t0, data.value
