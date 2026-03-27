# %%
import matplotlib.pyplot as plt
import numpy as np
from analysis_techniques.ligonoise import LIGOEvent
from analysis_techniques.welch_method import welch
from scipy.signal.windows import blackman
import scipy.signal as sig

# %%
dur = 200
N = dur * 4096
EventObject = LIGOEvent(dur)
# %%
print(f"This dataset contains {N} datapoints.")
data = EventObject.get_data()
t0, dt = EventObject.get_time_vars()
print(f"This would create {len(data)//(256//2)} bins of len 256 at 50% overlap.")

# %%
seglen = 4096*10
overlap = 0.75
test_f, test_psd = sig.welch(data, 4096, "blackman", seglen, (int((seglen*overlap)//1)))
freqs, PSD = welch(data, 4096, blackman, seglen, overlap)
# %%
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
ax[0].tick_params("both", labelsize=22)
ax[0].plot(freqs, PSD, label="Custom Welch")
ax[0].plot(test_f, test_psd, label="SciPy Welch")
ax[0].loglog()
ax[0].set_xlabel(r"Frequency/$Hz$", fontsize=30)
ax[0].set_ylabel(r"Relative Power", fontsize=30)
ax[0].set_title("PSD from Welch's Method", fontsize=30, pad=35)
ax[0].legend(fontsize=22)

error = np.abs((PSD - test_psd)/test_psd)
ax[1].plot(freqs, error, label="Relative Error")
ax[1].tick_params("both", labelsize=22)
ax[1].yaxis.get_offset_text().set_fontsize(22)
ax[1].set_xscale("log")
ax[1].set_xlabel(r"Frequency/$Hz$", fontsize=30)
ax[1].set_ylabel(r"Relative Error", fontsize=30)
ax[1].set_title("Relative Error Against SciPy Function", fontsize=30)
ax[1].axvline(35, linestyle="--", color="k", alpha=0.7)
ax[1].axvline(350, linestyle="--", color="k", alpha=0.7)
ax[1].axvspan(35, 350, color='green', alpha=0.1, label="Region of Interest")
ax[1].legend(fontsize=22, loc="upper right")
plt.suptitle("Implementated Welch Algorithm compared to SciPy", fontsize=35, fontweight="bold")
plt.tight_layout()
plt.show()

# %%
