# %%
import matplotlib.pyplot as plt
import numpy as np
from librosa import cqt
from librosa.display import specshow
from scipy.signal.windows import tukey
from analysis_techniques.ligonoise import LIGOEvent
from analysis_techniques.welch_method import welch
from analysis_techniques.data_processing import whiten


# %%
EventObject = LIGOEvent(200, 7)
strain = EventObject.get_data()
t0, dt = EventObject.get_time_vars()
print(t0)

# %%
fs = 1/dt.value
seglen = int(fs)*4
freqs, PSD = welch(strain, fs, tukey, seglen, overlap=0.75)
L = strain.shape[0]//2
step = int(fs*2)
segment = strain[L-step:L+step]

# %%
whitened_data = whiten(segment, fs, freqs, PSD, tukey)
window = tukey(whitened_data.shape[0])
windowed = whitened_data * window
# %%
# Plotting stats
freq = 4096
fmin = 32
bins_per_octave = 8
n_bins = int(np.floor(np.log2(freq/2/fmin) * bins_per_octave))
hop_length = 8


C = np.abs(cqt(windowed, sr=freq, hop_length=hop_length,
               fmin=fmin, n_bins=n_bins,
               bins_per_octave=bins_per_octave))
C = np.power(C, 2)
fig, ax = plt.subplots()
img = specshow(C, sr=freq, x_axis="time",
               y_axis="cqt_hz", ax=ax,
               fmin=fmin, hop_length=hop_length,
               bins_per_octave=bins_per_octave)
plt.show()
# %%
