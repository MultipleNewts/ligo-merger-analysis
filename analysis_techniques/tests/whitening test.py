# %%
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from analysis_techniques.ligonoise import LIGOEvent
from scipy.signal.windows import blackman
from scipy.signal import butter, sosfilt

data = LIGOEvent(10)

# %%
test = TimeSeries.fetch_open_data("H1", 1388811889.8-1, 1388811889.8+1)

# %%
data.data.asd().plot()
# %%
event = data.get_data()
t0, dt = data.get_time_vars()
windowed_data = blackman(len(event))*event

FFT_data = np.fft.rfft(windowed_data)
PSD = (1/len(event))*(np.abs(FFT_data)**2)
freqs = np.fft.rfftfreq(len(event), dt)
ASD = np.sqrt(FFT_data*FFT_data)

# %%
plt.plot(freqs, ASD)
plt.loglog()
plt.xlim([10, 2000])
plt.xlabel(r"Frequency/$Hz$")
plt.ylabel(r"ASD (strain/rtHz)")
plt.title("LIGO Strain data near an Event")

# %%
test.asd().plot()
plt.xlim([10, 2000])
# %%
fs = 4096
incr = fs/4
tot = fs*10
segment = event[int(tot/2-incr):int(tot/2+incr)]
interp_ASD = np.interp(np.fft.rfftfreq(len(segment), fs), np.fft.rfftfreq(len(event), fs), ASD)
FFT_Data = np.fft.rfft(segment)/(interp_ASD/dt/2)
FFT_freqs = np.fft.rfftfreq(len(segment), dt)
# FFT_Data = np.fft.rfft(event)[3500:4500]/ASD
# bandpass = butter(8, [35, 350], "bandpass", fs=4096, output="sos")
whitened_data = np.fft.irfft(FFT_Data)
# filtered = sosfilt(bandpass, whitened_data)
# T = 2*np.arange(N)/N
# plt.plot(filtered)
plt.plot(whitened_data)
plt.xlabel(r"Time/$s$")
plt.ylabel(r"Strain")
plt.title("Whitened Data")
# %%
len(event)

# %%
fs = 4096
asd_freqs = np.fft.rfftfreq(len(event), dt)
segment = event[int(40960/2) - int(incr) : int(40960/2) + int(incr)]

w = blackman(len(segment))
segment_ft = np.fft.rfft(segment)
segment_freq = np.fft.rfftfreq(len(segment), dt)
# interp_ASD = np.interp(segment_freq, asd_freqs, ASD)
# whitened_ft = segment_ft / (interp_ASD) / 2048
interp_PSD = np.interp(segment_freq, asd_freqs, PSD)
whitened_ft = segment_ft / np.sqrt(interp_PSD) / 2048
whitened_data = np.fft.irfft(whitened_ft)
plt.plot(whitened_data)
plt.show()

# %%
white_data = test.whiten()
bp_data = white_data.bandpass(30, 400)
fig3 = bp_data.plot()

# %%
