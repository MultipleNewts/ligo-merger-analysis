# %%
import scipy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = Path(__file__).parent/ "./Hello World.wav"

samplerate, sample = scipy.io.wavfile.read(path)
sample = sample[30000:90000]
times = np.linspace(0, 60000, len(sample))/samplerate
# plt.plot(times, sample)
plt.specgram(sample[:, 1], Fs=samplerate, cmap="plasma")
plt.ylim([0, 10000])
plt.show()

# %%
