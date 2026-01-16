# %%
import numpy as np
import matplotlib.pyplot as plt
import time


# Configure Random
time_seed = time.time_ns()
rnd_object = np.random.default_rng(seed=time_seed)

# Define Parameters
t_start = 0
t_end = 200
sample_rate = 30
N = np.int32((t_end-t_start) * sample_rate)

noise_mean = 0
noise_std = 1

signal_wavenumber = 2 * np.pi / 0.3
signal_width = 1
signal_amplitude = 1


# Signal Model
def model(t, t0=0):
    return (signal_amplitude *
            np.exp((-np.power((t-t0)/signal_width, 2))) *
            np.sin((t-t0)*signal_wavenumber))


# Generate Data
times = np.linspace(t_start, t_end, N, endpoint=True)
noise = rnd_object.normal(noise_mean, noise_std, N)

t0 = rnd_object.uniform(t_start, t_end)
signal = model(times, t0)

data = signal + noise

# %%
# Display
plt.plot(times, data, label="data")
plt.plot(times, signal, label="signal")
plt.legend(loc="upper right")
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("Basic Model")
plt.show()

# %%
log_likelihood = np.zeros(N)
for i in range(0, N):
    if (i % 1000 == 0):
        print(f"{(i/N*100):.3}%")
    log_likelihood[i] = -np.sum(np.power((data-model(times, times[i])), 2))
    # log_likelihood[i] = -np.sum(np.power((noise), 2))

MLE = np.argmax(log_likelihood)

plt.plot(times, log_likelihood, label="Log-Likelihood")
plt.axvline(MLE/N*t_end, c="red", linestyle="--", alpha=0.7, label="MLE")
plt.legend()
plt.show()


# %%
print(10000 / (t_end-t_start))
# %%
