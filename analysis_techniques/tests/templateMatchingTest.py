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

plt.plot(template.times[:-1], template.hp[:-1])
plt.show()
# %%
whitened_template_hp = whiten(template.hp[:-1], 4096, freqs, PSD)
plt.plot(template.times[:-1], whitened_template_hp)

# %%
template.times.shape[0]

# %%
