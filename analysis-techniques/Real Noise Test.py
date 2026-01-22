# %%
from gwpy.table import EventTable
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
import numpy as np


# get GWTC-4.0 events table: [0] is most recent, [128] is least recent from GWTC-4.0
events_gwtc4 = EventTable.fetch_open_data("GWTC-4.0")

# Find GPS times of two consecutive LIGO-H1 events
# Time between events: 185536.1 secs || 200 second buffer (=185336.1 secs of noise) to avoid signals
noise_start = (event_gps(events_gwtc4[1][0]) + 100)
noise_end = (event_gps(events_gwtc4[0][0]) - 100)

# Generate random location within noise data
max_length = noise_end - noise_start
loc = np.random.randint(0, (max_length-1000))
dur = 1000
gps_loc = noise_start + loc

# Fetch data from LIGO open dataset
data = TimeSeries.fetch_open_data("H1", gps_loc, gps_loc+dur)

# Plot data
plot = data.plot(
    title="LIGO Hanford Observatory: Noise",
    ylabel="Strain amplitude",
    color="gwpy:ligo-livingston",
    epoch=gps_loc,
)
plot.show()

# %%
