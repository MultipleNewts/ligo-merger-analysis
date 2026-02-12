# %%
import numpy as np
import matplotlib.pyplot as plt
from analysis_techniques.ligonoise import LIGONoise
from scipy.signal import windows
from scipy.signal import welch
from scipy.signal import butter, filtfilt
from gwpy.timeseries import TimeSeries
# %%


# %%
t0 = 1388811889.8
detector = "H1"
int_t0 = int(t0)
length = 200
strain = TimeSeries.fetch_open_data(detector, int_t0-int(length/2), int_t0+int(length/2)).value



# %%
# PLOT STRAIN
times = np.linspace(-int(length/2), int(length/2), strain.shape[0])
fig, axes = plt.subplots(figsize=(24,6))
axes.plot(times, strain)

# %%
# FFT of entire strain (don't use bad)
freq_bins = np.fft.rfftfreq(strain.shape[0], 1/4096)
ft = np.fft.rfft(strain*windows.hann(strain.shape[0]))
plt.loglog(freq_bins, np.abs(ft)* np.sqrt(2/length))
plt.xlim(10, 2048)
#plt.ylim(1e-24, 1e-19)
plt.show()

# %%
# CALC PSD FROM BIG DATA
toTransform = strain[:]
freq = 4096
psdfreqBins, PSD = welch(toTransform, freq, nperseg=4*freq, detrend="linear")
ASD = np.sqrt(PSD)
plt.loglog(psdfreqBins, ASD)
plt.xlim(10, 2048)
plt.ylim(1e-24, 1e-19)


# %%
# SEGMENT DATA TO WHAT WE WANT
int_center = int(strain.shape[0]/2)
segment_length = 0.5
segment = strain[int_center-int(segment_length/2 * freq):int_center+int(segment_length/2 * freq)]
times = np.linspace(-segment_length/2, segment_length/2, segment.shape[0])
plt.plot(times, segment, linewidth=1)
plt.show()


# %%
# DE LINEARISE SEGMENT
fit = np.polyfit(times, segment, 1)
print(fit)
delinearised_segment = segment - (fit[0]*times + fit[1])
# %%
# TAKE SEGMENT FFT
window = windows.hann(segment.shape[0])
windowed_segment = delinearised_segment * window
segment_ft = np.fft.rfft(windowed_segment)
segment_ft_freq_bins = np.fft.rfftfreq(segment.shape[0], 1/freq)
plt.loglog(segment_ft_freq_bins, np.abs(segment_ft), linewidth=1)
plt.xlim(10, 2048)
plt.show()


# %%
# CREATE INTERPRETTED ASD
inter_ASD = np.sqrt(np.interp(segment_ft_freq_bins, psdfreqBins, PSD))
plt.loglog(segment_ft_freq_bins, inter_ASD*freq*2)

plt.xlim(10, 1024)
# %%
# WHITEN DATA FT
whitened_ft = segment_ft
plt.loglog(segment_ft_freq_bins, np.abs(whitened_ft))
plt.show()


# %%
# INVERSE WHITENED DATA FT
whitened_data = np.fft.irfft(whitened_ft)
plt.plot(times, whitened_data, linewidth=1)
plt.ylim(-4,4)
# %%

# %%
