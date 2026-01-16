# %%
import numpy as np
# import matplotlib.pyplot as plt
import time


# Configure Random
time_seed = time.time_ns()
rnd_object = np.random.default_rng(seed=time_seed)


class WhiteNoise:
    def __init__(self, num, mean, std):
        self.mean = mean
        self.std = std
        self.gen_noise(num)

    def gen_noise(self, num):
        self.array = rnd_object.normal(self.mean, self.std, num)

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

    def get_noise(self):
        return self.array
