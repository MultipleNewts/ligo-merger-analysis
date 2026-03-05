# %%
import numpy as np
import matplotlib.pyplot as plt
from analysis_techniques.ligonoise import LIGOEvent
from analysis_techniques.welch_method import welch
from analysis_techniques.data_processing import whiten, bandpass
from scipy.signal.windows import tukey


# %%
EventObject = LIGOEvent(200, 7)
strain = EventObject.get_data()
t0, dt = EventObject.get_time_vars()
# %%
fs = 1/dt.value
seglen = int(fs)*4
freqs, PSD = welch(strain, fs, tukey, seglen, overlap=0.75)
L = strain.shape[0]//2
step = int(4096*2)
segment = strain[L-step:L+step]

# %%
whitened_data = whiten(segment, fs, freqs, PSD, tukey)
bandpassed_data = bandpass(whitened_data, 4096, order=8)
print(np.mean(whitened_data))
print(np.std(whitened_data))
times = np.linspace(-step*dt.value, step*dt.value, segment.shape[0])
plt.plot(times, bandpassed_data)
plt.xlabel(r"Time $(s)$")
plt.ylabel("Strain")
plt.title("Time Domain Whitened, Bandpassed Data Signal")
plt.show()
# %%
plt.figure(figsize=(12, 5))
bp_data = bandpass(whitened_data, fs, order=8)
plt.plot(times, bp_data, lw=1)
plt.xlabel(r"Time $(s)$")
plt.xlim((-0.25, 0.25))
plt.ylabel(r"Strain /$\sigma$")
plt.title("Bandpassed & Whitened Data Signal")
plt.show()

# %%
