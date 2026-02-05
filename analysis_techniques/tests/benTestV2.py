# %%
import numpy as np
import matplotlib.pyplot as plt
from analysis_techniques.ligonoise import LIGONoise
from scipy.signal import windows
from scipy.signal import welch
from scipy.signal import butter, filtfilt
from gwpy.timeseries import TimeSeries

# %%
t0 = 1387620938.3
detector = "H1"
int_t0 = t0
length = 200
strain = TimeSeries.fetch_open_data(detector, int_t0-int(length/2), int_t0+int(length/2)).value

# %%
# Calculate PSD of series
freq = 4096
freqs, PSD = welch(strain, fs=4096, window=("tukey", 0.25), nperseg=4*freq, noverlap=2*freq, detrend="linear", scaling="density")
ASD = np.sqrt(PSD)
plt.loglog(freqs, ASD)
plt.xlim(10, 2048)
plt.ylim(1e-24, 1e-19)
plt.xlabel("Frequency / hz")
plt.ylabel("ASD / strain/ sqrt(hz)")
plt.title("Welch ASD")
plt.grid(True)
plt.show()


# %%
int_center = int(strain.shape[0]/2)
half_length = 4
segment = strain[int_center - int(half_length*freq):int_center + int(half_length*freq)]
times = np.linspace(-half_length, half_length, segment.shape[0])


fit = np.polyfit(times, segment, 1)


plt.plot(times, segment)
plt.xlabel("Time / s")
plt.ylabel("Strain")
plt.title("Segment of data")
plt.show()
# %%
# Process the segment

# Remove DC signal
fit = np.polyfit(times, segment, 1)
segment_delinearised = segment - (fit[0] * times + fit[1])

window = windows.tukey(segment.shape[0])
power_correction = np.sum(np.power(window, 2)) / window.shape[0]

windowed_segment = window * segment_delinearised

segment_ft = np.fft.rfft(windowed_segment)
segment_freqs = np.fft.rfftfreq(segment.shape[0], 1/freq)

plt.loglog(segment_freqs, np.abs(segment_ft), linewidth=1)
plt.xlabel("Frequency / Hz")
plt.ylabel("Fourier amplitude")
plt.title("Fourier transform of pre-processed segment")
plt.show()


# %%
# Compare to ASD
interpretted_PSD = np.interp(segment_freqs, freqs, PSD)
plt.loglog(segment_freqs, np.sqrt((interpretted_PSD * power_correction) / (2 /freq)), linewidth=1)
plt.loglog(segment_freqs, np.abs(segment_ft), linewidth=1)
plt.xlabel("Frequency / Hz")
plt.ylabel("Fourier amplitude")
plt.title("Fourier transform of pre-processed segment")
plt.show()
# %%
# Give it the ASD

whitened_ft = segment_ft * np.sqrt(1 / power_correction / freq) / np.sqrt(interpretted_PSD)
plt.loglog(segment_freqs, whitened_ft)
plt.show()
# %%
whitened = np.fft.irfft(whitened_ft)
plt.plot(times, whitened)
plt.show()

# %%
np.mean(whitened)
np.var(whitened)
# %%

# %%
# Band pass
plt.specgram(whitened)
plt.show()
# %%

lowFreq = 35
highFreq = 350
order = 4
b, a = butter(order, [lowFreq, highFreq], btype="band", fs=freq)
whitened_bp = filtfilt(b, a, whitened)
plt.figure(figsize=(24,10))
plt.plot(times, whitened_bp)
plt.xlim((-0.2, 0.2))
plt.show()
# %%
np.var(whitened_bp)
#np.mean(whitened_bp)
# %%
plt.specgram(whitened_bp, freq)
# %%
