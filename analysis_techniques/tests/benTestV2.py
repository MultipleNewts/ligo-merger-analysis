# %%
import numpy as np
import matplotlib.pyplot as plt
from analysis_techniques.ligonoise import LIGOEvent
from analysis_techniques.welch_method import welch as custom_welch
from scipy.signal import windows
from scipy.signal import welch
from scipy.signal import butter, filtfilt

# %%
EventObject = LIGOEvent(200, 7)  # Index 0 for newest, index 7 for largest
strain = EventObject.get_data()
t0, dt = EventObject.get_time_vars()

# %%
# Calculate PSD of series
freq = 1/dt.value
freqs, PSD = welch(
                strain,
                fs=freq,
                window=("tukey", 0.25),
                nperseg=10*freq,
                noverlap=2*freq,
                detrend="linear",
                scaling="density"
            )
ASD = np.sqrt(PSD)
plt.loglog(freqs, ASD)
plt.xlim(10, 2048)
plt.ylim(1e-24, 1e-19)
plt.xlabel(r"Frequency $(Hz)$")
plt.ylabel(r"ASD $\left(\frac{strain}{\sqrt{hz}}\right)$")
plt.title("SciPy Welch ASD")
plt.grid(True)
plt.show()


# %%
# Use our own welch
window_fn = windows.tukey
seglen = int(freq * 10)
overlap = 0.75
freqs, PSD = custom_welch(strain, freq, window_fn, seglen, overlap)
ASD = np.sqrt(PSD)
plt.loglog(freqs, ASD)
plt.xlim(10, 2048)
plt.ylim(1e-24, 1e-19)
plt.xlabel(r"Frequency $(Hz)$")
plt.ylabel(r"ASD $\left(\frac{strain}{\sqrt{hz}}\right)$")
plt.title("Custom Welch ASD")
plt.grid(True)
plt.show()

# %%
# Get Segment
int_center = int(strain.shape[0]/2)
half_length = 4
segment = strain[int_center - int(half_length*freq):int_center + int(half_length*freq)]
times = np.linspace(-half_length, half_length, segment.shape[0])


fit = np.polyfit(times, segment, 1)

plt.figure(figsize=(24, 10))
plt.plot(times, segment)
plt.xlabel("Time / s")
plt.ylabel("Strain")
plt.title("Segment of data")
plt.xlim(-0.2, 0.2)
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
NoiseASD_data = np.sqrt((interpretted_PSD * power_correction) / (2/freq))
plt.loglog(segment_freqs, NoiseASD_data, linewidth=1, label="Noise ASD")
plt.loglog(segment_freqs, np.abs(segment_ft), linewidth=1, label="Data ASD")
plt.xlabel(r"Frequency $(Hz)$")
plt.ylabel("Fourier Amplitude")
plt.title("Fourier Transform of Pre-Processed Segment")
plt.legend()
plt.show()
# %%
# Factor out the Noise ASD

whitened_ft = segment_ft * np.sqrt(2 / power_correction / freq) / np.sqrt(interpretted_PSD)
plt.loglog(segment_freqs, np.abs(whitened_ft))
plt.xlabel(r"Frequency $(Hz)$")
plt.ylabel("Absolute Fourier Amplitude")
plt.title("Frequency Domain Whitened Data Signal")
plt.show()
# %%
whitened = np.fft.irfft(whitened_ft)
plt.plot(times, whitened)
plt.xlabel(r"Time $(s)$")
plt.ylabel("Strain")
plt.title("Time Domain Whitened Data Signal")
plt.show()

# %%
print(f"The bandpassed mean is: {np.mean(whitened)}\n"
      f"The bandpassed variance is: {np.var(whitened)}")
# %%
# Spectrogram
plt.specgram(whitened,
             Fs=freq, scale="linear",
             NFFT=512, noverlap=300, detrend="mean")
# plt.specgram(whitened[int(segment.shape[0]//2-freq/2):int(segment.shape[0]//2 + freq/2)],
#              Fs=freq, scale="linear",
#              NFFT=64, noverlap=50, detrend="mean")
plt.show()
# %%
# Bandpass

lowFreq = 35
highFreq = 350
order = 4
b, a = butter(order, [lowFreq, highFreq], btype="band", fs=freq)
whitened_bp = filtfilt(b, a, whitened)
plt.figure(figsize=(24, 10))
plt.plot(times, whitened_bp)
plt.xlim((-0.2, 0.2))
plt.xlabel(r"Time $(s)$")
plt.ylabel(r"Strain Amplitude")
plt.title("Blackhole Merger Signal")
plt.show()
# %%
print(f"The bandpassed mean is: {np.mean(whitened_bp)}\n"
      f"The bandpassed variance is: {np.var(whitened_bp)}")
# %%
segment_data = whitened_bp[int(segment.shape[0]//2-freq/2):int(segment.shape[0]//2 + freq/2)]
plt.specgram(segment_data, Fs=freq, scale="linear")
plt.show()
# %%
