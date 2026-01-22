# %%
from gwpy.table import EventTable
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
import numpy as np


class LIGONoise:
    def __init__(self, dur, loc=0, event_index=0):
        """Returns a dataset of noise from the LIGO-H1 GWTC-4.0 Catalogue

        Parameters
        ----------
        dur : `int`
            the duration of the signal

        loc : `int`
            the offset of the noise segment from the start of the noise sample
            Note: one must be cautious not to choose a location that overlaps
             with a succeeding events, generally keeping ``loc < 100000`` is reasonable.

        event_index : `int`
            the index of the event in the GWTC-4.0 catalogue
            ``[0]`` (also default) represents the most recent event: GW240109_050431
        """
        events_gwtc4 = EventTable.fetch_open_data("GWTC-4.0")
        noise_start = (event_gps(events_gwtc4[event_index][0]) + 100)
        self.loc = noise_start + loc
        self.data = TimeSeries.fetch_open_data("H1", self.loc, self.loc+dur)

    def get_noise(self):
        """Returns data array of fetched noise"""
        return self.data.value

    def get_time_vars(self):
        """Returns initial GPS time and time spacing between array elements

        Returns
        -------
        t0 : `float`
            the GPS time that the data window begins at, in seconds

        dt : `float`
            the time spacing between each data reading, in seconds
        """
        return self.data.t0, self.data.dt

    def segment(self, num=2):
        """Splits noise dataset into ``[num]`` equal segments

        Parameters
        ----------
        num : `int`, optional
            the number of equal parts for the data to be split into,
            defaults to 2

        Returns
        -------
        data_segments : `array`
            an array containing the equal-sized segments of the original noise array
        """
        len_each = len(self.data)//num
        segments = np.array_split(self.data, num)
        temp = []
        for segment in segments:
            if len(segment) != len_each:
                temp.append(segment[:len_each-1])
            else:
                temp.append(segment)
        return temp

    def plot(self):
        """
        Uses ``gwpy.timeseries.TimeSeries`` built-in methods to display the noise data graphically
        """
        plot = self.data.plot(
            title="LIGO Hanford Observatory: Noise",
            ylabel="Strain amplitude",
            color="gwpy:ligo-hanford",
            epoch=self.loc,
        )
        plot.show()
