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
sample_rate = 480
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

t0_index = rnd_object.integers(0, N)
t0 = t_start + t0_index/sample_rate


signal = model(times, t0)

data = signal + noise

# %%
# Display data
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

# %%
# Plot the log likelihood graph

MLE_index = np.argmax(log_likelihood)
MLE = log_likelihood[MLE_index]

log_likelihood_average = np.average(log_likelihood)
log_likelihood_MLE_diff = MLE - log_likelihood_average

print(f"MLE: {MLE}")
print(f"Log likelihood average: {log_likelihood_average}")

plt.title("Log likelihood of finding signal at time, $t_0$")
plt.xlabel("$t_0$ / s")
plt.ylabel("loglikelihood")
plt.plot(times, log_likelihood, label="Log-Likelihood", c="paleturquoise")
plt.axvline(t0, c="indigo", linestyle="--", alpha=0.5, label="$t_0$")
plt.axvline(MLE_index/N*t_end, c="red", linestyle="--", alpha=0.5, label="MLE")
plt.axhline(log_likelihood_average, c="darkgreen", linestyle="--")
plt.grid()
plt.legend()
plt.show()


# %%
# Normalise the log likelihood data


log_like_diffs = log_likelihood - log_likelihood_average
log_like_diffs_normalised = log_like_diffs / \
    (log_likelihood[t0_index] - log_likelihood_average)

log_like_diffs_normalised = \
    np.where(log_like_diffs_normalised > 1, 1, log_like_diffs_normalised)

plt.plot(times, log_like_diffs_normalised, c="paleturquoise")
plt.axvline(t0, c="indigo", linestyle="--", alpha=0.5, label="$t_0$")
plt.axvline(MLE_index/N*t_end, c="red", linestyle="--", alpha=0.5, label="MLE")
plt.legend()
plt.show()
# %%
# Create probability distribution

bin_count = 100 + 2
bins = np.linspace(-1, 1.01, bin_count)

binned_data = np.digitize(log_like_diffs_normalised, bins)
data_distribution = np.bincount(binned_data, minlength=bin_count) / N

print(data_distribution.shape[0], bins.shape[0])
std_dev = 2 / bin_count * np.std(binned_data)
print(std_dev)
# %%
plt.plot(bins, data_distribution, label="data")
plt.axvline(std_dev, ls="--", c="violet", label=r"$1 \sigma$")
plt.axvline(-std_dev, ls="--", c="violet")
plt.axvline(2*std_dev, ls="--", c="navajowhite", label=r"$2 \sigma$")
plt.axvline(-2*std_dev, ls="--", c="navajowhite")
plt.legend()
plt.title("Log likelihood deviations from average.")
plt.xlabel("Fraction of the log likelihood for $t=t_0$")
plt.ylabel("Probability Density")
plt.show()
# %%
# Note, only do one of these at a time
run_15 = data_distribution
# %%
run_30 = data_distribution
# %%
run_60 = data_distribution
# %%
run_120 = data_distribution
# %%
run_240 = data_distribution

# %%
run_480 = data_distribution

# %%

plt.plot(bins, run_15, label="15")
plt.plot(bins, run_30, label="30")
plt.plot(bins, run_60, label="60")
plt.plot(bins, run_120, label="120")
plt.plot(bins, run_240, label="240")
plt.plot(bins, run_480, label="480")
plt.legend(title=r"Sample rate / hz")
plt.title("Log likelihood deviations from average.")
plt.xlabel("Fraction of the log likelihood for $t=t_0$")
plt.yscale("log")
plt.ylabel("Probability Density")
plt.show()
# %%
