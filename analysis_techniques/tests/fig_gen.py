# %%
from analysis_techniques.templateImporting import Templates
from analysis_techniques.ligonoise import LIGOEvent
from analysis_techniques.event_detector_v2 import detect_events_v2, find_best_data_v2
from analysis_techniques.welch_method import welch
from analysis_techniques.data_processing import whiten, bandpass, normalise
from librosa import cqt
from librosa.display import specshow
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
import numpy as np
# %%
dataObject = LIGOEvent(200, 7)
# %%
dataObject2 = LIGOEvent(200, 3)
# %%
data = dataObject.data
t0, dt = dataObject.get_time_vars()
L = len(data)
times = np.linspace(0, L*dt, L)
half = (L*dt/2).value
lims = [half-0.75, half+0.75]
print(lims)
# %%
et = 1388422190.6
print(et-100)

# %%
def proper_plt(X, Y, label, title, xlab, ylab, lims=None, log=False, ft=30, custom=False):
    fig, ax = plt.subplots(1, 1, figsize=(24, 10), tight_layout=True)
    ax.plot(X, Y, label=label)
    ax.set_title(title, fontsize=int(ft*1.33))
    ax.set_xlabel(xlab, fontsize=int(ft*1.167))
    ax.set_ylabel(ylab, fontsize=int(ft*1.167))
    ax.tick_params("both", length=8, width=3, labelsize=ft)
    ax.tick_params("both", which="minor", length=5, width=3, labelsize=ft)
    ax.yaxis.get_offset_text().set_fontsize(ft)
    ax.legend(fontsize=int(ft*1.167), loc="upper right")
    if lims is not None:
        ax.set_xlim(lims)
    if log is True:
        ax.loglog()
    if custom is True:
        ax.annotate("Inspiral", [99.97, 0], [99.94, 4], fontsize=35, arrowprops={"width": 3})
        ax.annotate("Merger Chirp", [100.003, -4.9], [100.03, -4], fontsize=35, arrowprops={"width": 3})
        ax.annotate("Ringdown", [100.02, 0], [100.035, 3], fontsize=35, arrowprops={"width": 3})
    ax.grid(True)
    plt.show()


# %%
proper_plt(
    times,
    data,
    "Raw Data",
    "Raw Data from LIGO Event GW231226_101520",
    r"Times $(s)$",
    r"Strain",
    lims
)
# %%
freqs, psd = welch(data, 4096, tukey, 4*4096, overlap=0.5)
proper_plt(
    freqs,
    psd,
    "Welch PSD",
    "Welch Power Spectral Density of Event GW231226_101520",
    r"Frequency $(Hz)$",
    r"Power Spectral Density",
    log=True,
    ft=37
)
# %%
w_data = whiten(data, 4096, freqs, psd)

proper_plt(
    times,
    w_data,
    "Whitened Data",
    "Whitened Data from LIGO Event GW231226_101520",
    r"Times $(s)$",
    r"Standard Deviations $(\sigma)$",
    lims,
    ft=35
)
# %%
bp_data = bandpass(w_data, 4096)
# %%
proper_plt(
    times,
    bp_data,
    "Bandpassed Data",
    "Bandpassed Data from LIGO Event GW231226_101520",
    r"Times $(s)$",
    r"Standard Deviations $(\sigma)$",
    [half-0.15, half+0.15],
    ft=35,
    # custom=True
)
# %%
fig, ax = plt.subplots(1, 2, figsize=(19,7))
ax[0].plot(times, bp_data, label="Processed Data")
ax[0].set_xlim([half-0.15, half+0.15])
ax[0].set_xlabel(r"Time $(s)$", fontsize=22)
ax[0].set_ylabel(r"Standard Deviations $(\sigma)$", fontsize=22)
ax[0].set_title("Time-Domain Result", fontsize=25, pad=15)
ax[0].tick_params("both", labelsize=20)
ax[0].legend(fontsize=20, loc="upper right")

windowed = bp_data * tukey(bp_data.shape[0])
freq = 4096
fmin = 32
bins_per_octave = 8
n_bins = int(np.floor(np.log2(freq/2/fmin) * bins_per_octave))
hop_length = 8
C = np.abs(cqt(windowed, sr=freq, hop_length=hop_length,
               fmin=fmin, n_bins=n_bins,
               bins_per_octave=bins_per_octave))
img = specshow(C, sr=freq, x_axis="time",
               y_axis="cqt_hz", ax=ax[1],
               fmin=fmin, hop_length=hop_length,
               bins_per_octave=bins_per_octave, cmap="viridis")
ax[1].set_xlim(lims)
ax[1].set_ylim([32, 512])
ax[1].set_xticks([99.25, 99.50, 99.75, 100.0, 100.25, 100.50, 100.75], labels=["", "99.50", "99.75", "100.0", "100.25", "100.50", "100.75"], fontsize=20)
ax[1].tick_params(axis="y", labelsize=20)
ax[1].set_ylabel(r"Frequency $(Hz)$", fontsize=25)
ax[1].set_xlabel(r"Time $(s)$", fontsize=25)
ax[1].set_title("CQT Spectrogram Result", fontsize=25, pad=15)
plt.suptitle("Processed Results for Event GW240104_164932", fontsize=28, fontweight="bold")
cb = fig.colorbar(img, ax=ax[1])
cb.set_label("Normalised Energy", fontsize=30)
cb.ax.tick_params(labelsize=25)
plt.tight_layout()
plt.show()
# %%
wd = data.whiten().value
error = np.abs((wd-w_data))
fig, ax = plt.subplots(1, 2, figsize=(17,7))
ax[0].plot(times, w_data, label="Custom Function")
ax[0].plot(times, wd, label="GWpy Function")
ax[0].set_xlim([half-0.15, half+0.15])
ax[0].set_xlabel(r"Time $(s)$", fontsize=22)
ax[0].set_ylabel(r"Standard Deviations $(\sigma)$", fontsize=22)
ax[0].set_title("Whitened Data", fontsize=25, pad=15)
ax[0].tick_params("both", labelsize=20)
ax[0].legend(fontsize=20, loc="upper right")

ax[1].plot(times, error, label="Absolute Error")
ax[1].set_xlim([half-0.15, half+0.15])
ax[1].set_ylim([-0.1, 1.25])
ax[1].set_xlabel(r"Time $(s)$", fontsize=22)
ax[1].set_ylabel(r"Standard Deviations $(\sigma)$", fontsize=22)
ax[1].set_title("Absolute Error with GWpy Function", fontsize=25, pad=15)
ax[1].tick_params("both", labelsize=20)
avg = np.mean(error)
ax[1].axhline(avg, linestyle="--", color="r", label=(f"Average Error: {avg:.2f}"+r"$\sigma$"))
ax[1].axhline(1, linestyle="--", color="k", alpha=0.7, label=r"$1\sigma$")
ax[1].legend(fontsize=20, loc="upper right")
plt.suptitle("Implemented Whitening compared to GWpy", fontweight="bold", fontsize=27)
plt.tight_layout()
plt.show()
# %%
proper_plt(
    times,
    error,
    "Bandpassed Data",
    "Bandpassed Data from LIGO Event GW231226_101520",
    r"Times $(s)$",
    r"Standard Deviations $(\sigma)$",
    [half-0.15, half+0.15],
    ft=35,
    # log=True
)
# %%









# %%
# CQT stuff
windowed_1 = bp_data * tukey(bp_data.shape[0])
# %%
freq = 4096
fmin = 32
bins_per_octave = 8
n_bins = int(np.floor(np.log2(freq/2/fmin) * bins_per_octave))
hop_length = 8

window = tukey(bp_data.shape[0])
# windowed = whitened_data * window
windowed = bp_data * window
# %%
C = np.abs(cqt(windowed, sr=freq, hop_length=hop_length,
               fmin=fmin, n_bins=n_bins,
               bins_per_octave=bins_per_octave))
fig, ax = plt.subplots(1, 2, layout="constrained", figsize=(17,10))
img = specshow(C, sr=freq, x_axis="time",
               y_axis="cqt_hz", ax=ax[0],
               fmin=fmin, hop_length=hop_length,
               bins_per_octave=bins_per_octave, cmap="viridis")

C2 = np.abs(cqt(windowed_1, sr=freq, hop_length=hop_length,
                fmin=fmin, n_bins=n_bins,
                bins_per_octave=bins_per_octave
                )
            )
img2 = specshow(C2, sr=freq, x_axis="time",
                y_axis="cqt_hz", ax=ax[1],
                fmin=fmin, hop_length=hop_length,
                bins_per_octave=bins_per_octave, cmap="viridis"
                )

ax[0].set_xlim(lims)
ax[1].set_xlim(lims)
ax[0].set_ylim([32, 512])
ax[1].set_ylim([32, 512])
ax[0].set_xticks([99.25, 99.50, 99.75, 100.0, 100.25, 100.50, 100.75], labels=["", "99.50", "99.75", "100.0", "100.25", "100.50", "100.75"], fontsize=20)
ax[1].set_xticks([99.25, 99.50, 99.75, 100.0, 100.25, 100.50, 100.75], labels=["", "99.50", "99.75", "100.0", "100.25", "100.50", "100.75"], fontsize=20)
ax[0].tick_params(axis="y", labelsize=20)
ax[1].tick_params(axis="y", labelsize=20)
ax[0].set_ylabel(r"Frequency $(Hz)$", fontsize=25)
ax[0].set_xlabel(r"Time $(s)$", fontsize=25)
ax[1].set_ylabel(r"Frequency $(Hz)$", fontsize=25)
ax[1].set_xlabel(r"Time $(s)$", fontsize=25)
cb = fig.colorbar(img, ax=ax[1])
cb.set_label("Normalised Energy", fontsize=30)
cb.ax.tick_params(labelsize=25)
ax[0].set_title("Event GW231226_101520 (High SNR)", fontsize=28, pad=20)
ax[1].set_title("Event GW240104_164932 (Low SNR)", fontsize=28, pad=20)
plt.suptitle("Constant Q-Transform for a High SNR and Low SNR Event", fontweight="bold", fontsize=35)
# plt.tight_layout()
plt.show()
# %%
