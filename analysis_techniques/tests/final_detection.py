# %%
from analysis_techniques.templateImporting import Templates
from analysis_techniques.ligonoise import LIGOEvent
from analysis_techniques.event_detector import detect_events, find_best_data
from scipy.signal.windows import tukey
# import matplotlib.pyplot as plt


# %%
EO3 = LIGOEvent(200, 3)
EO7 = LIGOEvent(200, 7)


# %%
def high_snr_search(event):
    # Load Large Dataset for Event 7 (High SNR)
    strain = event.get_data()
    t0, dt = event.get_time_vars()
    fs = 1/dt.value

    # Load Templates
    Template_Manager = Templates("templates/Event7FinalTemplates.json")
    print(f"Event 7 will use: {Template_Manager.template_count} templates")

    # Detect and Match Events for Event 7
    event_times, event_plots, ip_vals = detect_events(strain, Template_Manager, fs, tukey, 4, 0.5)

    # Output Best Matched Model, if event in data
    results = find_best_data(Template_Manager, event_times, event_plots, fs)
    return results


def low_snr_search(event):
    # Load Large Dataset for Event 3 (Low SNR)
    strain = event.get_data()
    t0, dt = event.get_time_vars()
    fs = 1/dt.value

    # Load Templates
    Template_Manager = Templates("templates/Event3FinalTemplates.json")
    print(f"Event 3 will use: {Template_Manager.template_count} templates")

    # Detect and Match Events for Event 3
    event_times, event_plots, ip_vals = detect_events(strain, Template_Manager, fs, tukey, 4, 0.5)

    # Output Best Matched Model, if event in data
    results = find_best_data(Template_Manager, event_times, event_plots, fs)
    return results, event_times, event_plots


# %%
results7 = high_snr_search(EO7)

# %%
results3, et, ep = low_snr_search(EO3)

# %%
