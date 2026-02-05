# %%
import matplotlib.pyplot as plt
import numpy as np
from analysis_techniques.ligonoise import LIGOEvent
from analysis_techniques.welch_method import welch
from scipy.signal.windows import blackman
import time

# %%
EventObject = LIGOEvent(200)
data = EventObject.get_data()


# %%
def run_welch(seglength, overlap):
    freqs, PSD = welch(data, 4096, blackman, seglength, overlap)
    return freqs, PSD


# %%
N = 10
overlaps = np.arange(N)/(1.11*N)
overlap_data = []
runtime = []
for overlap in overlaps:
    temp = time.time_ns()
    overlap_data.append(run_welch(4096*4, overlap)[1])
    runtime.append(time.time_ns()-temp)
freqs = np.fft.rfftfreq(4096*4, (1/4096))

fig, ax = plt.subplots(2, 1)
for overlap, dataset in zip(overlaps, overlap_data):
    ax[0].plot(freqs, dataset, label=f"{overlap} overlap")
ax[1].plot(overlaps, runtime, label="Time Taken")

ax[0].loglog()
ax[0].set_xlabel(r"Frequency/$Hz$")
ax[0].set_ylabel(r"Relative Power")
ax[0].set_title("PSD from Welch's Method")
ax[1].set_xlabel(r"Overlap Ratio")
ax[1].set_ylabel(r"Time Taken/$ns$")
ax[1].set_title("PSD from Welch's Method")
ax[1].legend()
plt.tight_layout()
plt.show()

# %%
N = 20
seglengths = np.arange(N, step=2)*4096
seg_data = []
runtime = []
for seg in seglengths:
    temp = time.time_ns()
    seg_data.append(run_welch(seg, 0.5))
    runtime.append(time.time_ns()-temp)

fig, ax = plt.subplots(2, 1)
for overlap, dataset in zip(overlaps, overlap_data):
    ax[0].plot(*dataset, label=f"{overlap} overlap")
ax[1].plot(seglengths, runtime, label="Time Taken")

ax[0].loglog()
ax[0].set_xlabel(r"Frequency/$Hz$")
ax[0].set_ylabel(r"Relative Power")
ax[0].set_title("PSD from Welch's Method")
ax[1].set_xlabel(r"Overlap Ratio")
ax[1].set_ylabel(r"Time Taken/$ns$")
ax[1].set_title("PSD from Welch's Method")
ax[1].legend()
plt.tight_layout()
plt.show()
