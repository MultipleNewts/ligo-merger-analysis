# %%
import numpy as np
import matplotlib.pyplot as plt
import time


# Configure Random
time_seed = time.time_ns()
rnd_object = np.random.default_rng(seed=time_seed)


# Define White Noise Class
class WhiteNoise:
    # Define required attributes
    def __init__(self, num, mean, std, samplerate=16000):
        self.label = "White Noise"
        self.mean = mean
        self.std = std
        self.samplerate = samplerate
        self.gen_noise(num)

    # Generate White Noise
    def gen_noise(self, num):
        self.array = rnd_object.normal(self.mean, self.std, num)

    # Plots Noise Data
    def plot_raw(self):
        plt.plot(self.array)
        plt.axhline(self.mean, color="r", linestyle="--", alpha=0.5, label=r"$\mu$")
        plt.axhline(self.mean+self.std, color="k", linestyle="--", alpha=0.5, label=r"$1\sigma$")
        plt.axhline(self.mean-self.std, color="k", linestyle="--", alpha=0.5)
        plt.xlabel("Datapoint")
        plt.ylabel("Value")
        plt.title(self.label)
        plt.legend(loc="upper right")
        plt.show()

    # Plots Distribution of Noise Data
    def plot_hist(self):
        plt.hist(self.array, color="deepskyblue", density=True)
        plt.xlabel("Value")
        plt.ylabel("Proportion")
        plt.title(self.label + " Distribution")
        plt.show()

    # Plots Spectrogram of Data
    def plot_spect(self):
        plt.specgram(self.array, Fs=self.samplerate, cmap="plasma")
        plt.xlabel("Value")
        plt.ylabel("Proportion")
        plt.title(self.label + " Distribution")
        plt.show()

    # Plots amplitudes of frequencies present
    def FFT_Freqs(self):
        freqs = np.fft.fftfreq(self.get_len(), 1/self.get_samplerate())
        FFT_data = np.fft.fft(self.array)
        freqs = self.trim_freqs(freqs)
        FFT_data = self.trim_freqs(FFT_data)
        plt.plot(freqs, 10*np.log10(np.abs(FFT_data)**2))
        # plt.xscale("log")
        plt.xlabel(r"Frequency/$Hz$")
        plt.ylabel("Amplitude/dB")
        plt.title(self.label + "FFT Frequency Distribution")
        plt.grid()
        plt.grid(which="minor")
        plt.show()

    # Trims FFT frequencies according to Nyquist-Shannon Sampling
    def trim_freqs(self, data):
        L = len(data)
        if L % 2 == 0:
            data = data[:int(L/2)]
        else:
            data = data[:int((L+1)/2)]
        return data

    # Get/Set Methods for Attributes
    def get_mean(self):
        return self.mean

    def set_mean(self, mean):
        self.mean = mean
        self.gen_noise(len(self.array))

    def get_std(self):
        return self.std

    def set_std(self, std):
        self.std = std
        self.gen_noise(len(self.array))

    def get_len(self):
        return len(self.array)

    def get_samplerate(self):
        return self.samplerate

    # Access Noise Data
    def get_noise(self):
        return self.array


# Define Pink (Fractional) Noise
class PinkNoise(WhiteNoise):
    def __init__(self, num, mean, std, samplerate=16000):
        super().__init__(num, mean, std, samplerate)
        self.label = "Pink Noise"

    def gen_noise(self, num):
        self.array = rnd_object.normal(self.mean, self.std, num)
        FFT_data = np.fft.fft(self.array)[1:]
        FFT_freqs = np.fft.fftfreq(self.get_len(), 1/self.get_samplerate())[1:]
        plt.plot(FFT_freqs, FFT_data)
        plt.show()
        print(".,.,.,")
        temp = []
        for val, f in zip(FFT_data, FFT_freqs):
            temp.append(val/np.emath.sqrt(f))
        plt.plot(FFT_freqs, temp)
        plt.show()
        print("<><><><><><")
        self.array = np.fft.ifft(temp)


WN = WhiteNoise(10000, 0, 1, 4000)
PN = PinkNoise(10000, 0, 1, 4000)
WN.FFT_Freqs()
PN.FFT_Freqs()
