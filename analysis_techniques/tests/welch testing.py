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
plt.plot(freqs, PSD, label="Custom Welch")
plt.plot(test_f, test_psd, label="SciPy Welch")
plt.loglog()
plt.xlabel(r"Frequency/$Hz$")
plt.ylabel(r"Relative Power")
plt.title("PSD from Welch's Method")
plt.legend()
plt.show()
# %%
error = np.abs((PSD - test_psd)/test_psd)
plt.plot(freqs, error, label="Absolute Error")
# plt.loglog()
plt.xscale("log")
plt.xlabel(r"Frequency/$Hz$")
plt.ylabel(r"Absolute Error")
plt.title("Absolute Error Benchmarked Against SciPy")
plt.legend()
plt.show()

# %%
