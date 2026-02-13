# %%
# %%
import numpy as np
import matplotlib.pyplot as plt
from analysis_techniques.ligonoise import LIGOEvent
from analysis_techniques.welch_method import welch
from analysis_techniques.data_processing import whiten, bandpass
from scipy.signal.windows import tukey

# %%
EventObject = LIGOEvent(200, 0)  # 7, 39 are big events
large_data = EventObject.get_data()
t0, dt = EventObject.get_time_vars()
fs = 1/dt.value
# %%
freqs, PSD = welch(large_data, fs, tukey, int(4*fs), overlap=0.75)

seglength = int(2*fs)
overlap = 0.5
step = int(seglength*(1 - overlap))
segments = []
events = []
processed = []
max_iter = int((len(large_data) - seglength)//step) + 1
for i in range(max_iter):
    index = int(i*step)
    segments.append(large_data[index:(index+seglength)])
for i, segment in enumerate(segments):
    whitened_data = whiten(segment, fs, freqs, PSD, tukey)
    bp_data = bandpass(whitened_data, fs, order=8)
    FFT = np.abs(np.fft.rfft(bp_data))
    processed.append(FFT)
    check = np.sum(np.where(FFT > 450, 1, 0))
    if check > 0:
        events.append(i)
print(events)
event_data = []
for event_idx in events:
    event_data.append(processed[event_idx])
for event in event_data:
    # L = len(event)
    # t = L/(2*fs)
    # times = np.linspace(-t, t, L)
    plt.plot(event)
    plt.xscale("log")
    # plt.xlim([-0.15, 0.05])
# %%
