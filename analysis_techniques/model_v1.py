# %%
import numpy as np
import matplotlib.pyplot as plt
import time
from analysis_techniques.ligonoise import LIGONoise


# Configure Random
time_seed = time.time_ns()
rnd_object = np.random.default_rng(seed=time_seed)

# Define Parameters
N = 10000  # Number of Datapoints
t_start = 0
t_end = 200


# Signal Model
def model(t, A, k, width, t0=0):
    return (A * np.exp((-np.power((t-t0)/width, 2))) * np.sin((t-t0)*k))


# Generate Noise
noise = LIGONoise(200).get_noise()

# %%
# Generate Signal
model_vars = (9e-19, 2*np.pi/10, 20)
times = np.linspace(t_start, t_end, N, endpoint=True)
buffer = (t_end - t_start) * 0.01
t0 = rnd_object.uniform(t_start+buffer, t_end-buffer)
signal = model(times, *model_vars, t0)

# Compute Data
data = signal + noise[:len(signal)]

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
    log_likelihood[i] = np.sum(np.power((data-model(times, *model_vars, times[i])), 2))

MLE = np.argmax(log_likelihood)

plt.plot(times, log_likelihood, label="Log-Likelihood")
plt.axvline(MLE/N*t_end, c="red", linestyle="--", alpha=0.7, label="MLE")
plt.legend()
plt.show()
