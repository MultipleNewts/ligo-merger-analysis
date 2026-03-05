# %%
import numpy as np
import matplotlib.pyplot as plt
from analysis_techniques.ligonoise import LIGOEvent
from analysis_techniques.welch_method import welch
from analysis_techniques.data_processing import whiten, bandpass
from scipy.signal.windows import tukey
from analysis_techniques.templateImporting import Templates

# %%
EventObject = LIGOEvent(200, 7)
strain = EventObject.get_data()
t0, dt = EventObject.get_time_vars()
fs = 1/dt.value
# %%
default = "templates/Event7Template.json"
BBHModel = "templates/Event7Template.json"
IMRPhenomXo4 = "templates/Event7TemplateIMRPhenomXo4.json"

Template_Manager = Templates(default)
template = Template_Manager.get_template(0)
model = template.hp[:]

model_times = template.times[:]
plt.plot(model)
# %%
fs = 1/dt.value
seglen = int(fs)*4
freqs, PSD = welch(strain, fs, tukey, seglen, overlap=0.75)
L = strain.shape[0]//2
step = 4096*2
segment = strain[L-step:L+step]
whitened_data = whiten(segment, fs, freqs, PSD, tukey)
bp_data = bandpass(whitened_data, fs, order=8)
times = np.linspace(-2, 2, segment.shape[0]) 
# %%
whitened_model = whiten(model, fs, freqs, PSD, tukey)
bp_model = bandpass(whitened_model, fs, order=8)

# %%
plt.plot(times, bp_data)
plt.show()
# %%
print(len(bp_data), len(bp_model))
L = len(bp_data)//2
split = len(bp_model)//2
bp_data_cut = bp_data[L-split:L+split]
times_cut = times[L-split:L+split]
print(len(bp_data_cut), len(bp_model))
# %%
plt.figure(figsize=(24, 10))
t0 =  -0.006772506772506792
plt.plot(model_times, bp_model)
plt.plot(times+t0, bp_data, alpha=0.75)
plt.xlim([-0.2, 0.2])
# %%
print(fs*t0)
# %%
np.linalg.norm([bp_data_cut[:-27], bp_model[27:]])
# %%

model_times = template.times[:]
# Good t0 ish = -0.0065
iters = 1000000
t0s = np.linspace(-0.5, 0.5, iters)
inner_prods = np.zeros(iters)
for i, t0 in enumerate(t0s):
    if i % 10000 == 0:
        print(i)
    interp_data = np.interp(model_times, times+t0, bp_data)
    inner_product = np.sum(interp_data * bp_model)
    inner_prods[i] = inner_product

# %%
optimal = t0s[np.argmax(inner_prods)]
print(optimal)
plt.plot(t0s, inner_prods)
plt.axvline(optimal, c="r", linestyle="--", alpha=0.2)
plt.show()

# %%
plt.figure(figsize=(24, 10))
plt.plot(model_times, bp_model)
plt.plot(times+optimal, bp_data)
t_low = np.min(model_times)
t_high = np.max(model_times)
plt.xlim((t_low, t_high))
plt.show()

# %%
