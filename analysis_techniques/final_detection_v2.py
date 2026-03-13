# %%
from analysis_techniques.templateImporting import Templates
from analysis_techniques.ligonoise import LIGOEvent
from analysis_techniques.event_detector_v2 import detect_events_v2, find_best_data_v2
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
import numpy as np
from analysis_techniques.welch_method import welch
from analysis_techniques.data_processing import whiten, bandpass

# %%
EO3 = LIGOEvent(200, 3)

# %%
strain = EO3.get_data()
t0, dt = EO3.get_time_vars()
fs = 1/dt.value

Template_Manager = Templates("templates/Event3FinalTemplates.json")

# %%
event_times, event_plots, products = detect_events_v2(strain, Template_Manager, fs, tukey, 4, 0.5)

# %%
find_best_data_v2(Template_Manager, event_times, event_plots)
# %%
