# %%
import matplotlib.pyplot as plt
import numpy as np
import time


# Config Random
time_seed = time.time_ns()
rnd_object = np.random.default_rng(seed=time_seed)


# Define parameters
N = 441000
t_start = 0
t_end = 10

noise_mean = 0
noise_std = 1


# Generate Noise
times = np.linspace(t_start, t_end, N, endpoint=True)
noise = rnd_object.normal(noise_mean, noise_std, N)


# %%
plt.plot(times, noise)
plt.ylabel("Displacement")
plt.xlabel("Time")
plt.title("White noise")
plt.show()


# %%
# Show Periodogram

freq_bins = np.fft.fftfreq(N, (t_end-t_start)/N)
noise_transform = np.fft.fft(noise)

noise_power = 10 * np.log10(np.power(np.abs(noise_transform), 2))

plt.bar(freq_bins, noise_power)
plt.xscale("log")

plt.xlim((0, np.max(freq_bins)))
plt.xlabel("Frequency / Hz")
plt.ylabel("Spectral Power / dB/Hz")
plt.title("White noise Periodogram")#
plt.show()
# %%
# Attempt as a filter