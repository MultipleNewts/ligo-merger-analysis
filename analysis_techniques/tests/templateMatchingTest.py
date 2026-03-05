# %%
import numpy as np
import matplotlib.pyplot as plt
from analysis_techniques.templateImporting import Templates
from analysis_techniques.welch_method import welch
from analysis_techniques.data_processing import whiten, bandpass
from analysis_techniques.ligonoise import LIGOEvent

# %%
event_object = LIGOEvent(200, 7)

strain = event_object.get_data()
t0, dt = event_object.get_time_vars()

# %%
templates = Templates("templates/Event7Template.json")
template = templates.get_template(0)
print(template.mass1, template.mass2, template.distance)

plt.plot(template.times, template.hp)
plt.title(
    f"Waveform: Mass1: {template.mass1}, "
    f"Mass2: {template.mass2}, "
    f"Distance: {template.distance}")
plt.xlabel("Time / s")
plt.ylabel("Strain")
plt.show()
# %%
freqs, PSD = welch(strain, 4096, seglength=int(4096)*4)
print(strain.shape[0])

# %%
shape = template.hp.shape[0]
T0s = np.arange(0, strain.shape[0]-10000)
filterValues = np.zeros(strain.shape[0]-10000)

for i in range(0, strain.shape[0]-10000):

    segmentT0Index = i
    segment = strain[segmentT0Index:segmentT0Index+shape]

    templateData = template.hp

    whitened_templateData = whiten(templateData, 4096, freqs, PSD, fourier_output=True)
    whitened_segment = whiten(segment, 4096, freqs, PSD, fourier_output=True)

    matched_filter = np.fft.irfft(whitened_segment * np.conj(whitened_templateData))
    filterValue = np.abs(np.sum(matched_filter))
    filterValues

 # %%
 # Prepare whitened template
template_times = template.times[:-1]
template_hp = template.hp[:-1]
whitened_template_hp = whiten(template_hp, 4096, freqs, PSD)

# Prepare whitened_segment
t0 = strain.shape[0] // 2
indexes = template_times.shape[0]
segment = strain[t0-indexes//2 : t0+indexes//2]
length = indexes*dt
segment_times = np.linspace(-length/2, length/2, indexes)



#plt.plot(template_times, whitened_template_hp)
plt.plot(segment_times, whitened_segment)
plt.show()
# %%
ting = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
indexes = 3
t0 = 5
ting[t0-3:t0+3]
# %%
