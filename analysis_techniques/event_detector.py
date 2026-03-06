# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from analysis_techniques.ligonoise import LIGOEvent
from analysis_techniques.welch_method import welch
from analysis_techniques.data_processing import whiten, bandpass
from analysis_techniques.templateImporting import Templates
import time
from cuda_flow.dot_product import inner_prod
from numba import cuda

# %%
# processes data (whitens and bandpasses)
def process_data(data, fs, freqs, PSD):
    whitened_data = whiten(data, fs, freqs, PSD, tukey)
    bp_data = bandpass(whitened_data, fs, order=8)
    return bp_data


# %%
def detect_events(full_data, template_bank, fs=4096, window_func=tukey, seg_dur=4, overlap=0.75, acceleration=False):
    # compute Welch PSD
    freqs, PSD = welch(strain, fs, window_func, int(seg_dur*fs), overlap=overlap)
    # compute test length
    test_length = int(4*fs)
    if test_length % 2 != 0:
        test_length -= 1
    test_index = test_length - 1
    # moving window setup
    overlap = 0.5
    step = int(test_length*(1 - overlap))
    segments = []
    processed = []

    # segments full dataset into segments
    max_iter = int((len(full_data) - test_length)//step) + 1
    for i in range(max_iter):
        index = int(i*step)
        segments.append(full_data[index:(index+test_length)])

    template_events = {}
    template_plots = {}
    # test each template from bank
    for i in range(template_bank.template_count):
        print(f"reached template {i}")
        # retrieves template from API
        template = template_bank.get_template(i)
        print(f"Masses {template.mass1}, {template.mass2}")
        # retrieves model data from template
        model = template.hp[:test_index]
        tmpl_split = int(len(model)//2)
        # processes model (whiten + bandpass) to ensure same scale as processed data
        proc_model = process_data(model, fs, freqs, PSD)
        # retrieves model timestamps (timestamp 0 centred on chirp)
        model_times = template.times[:test_index]
        events = []
        plots = []

        # iterates through every window segment in dataset
        for j, segment in enumerate(segments):
            if j % 10 == 0:
                print(f"Processing Segment {j}...")
            # processes each window
            proc_data = process_data(segment, fs, freqs, PSD)
            test_dur = test_length/fs
            dt = 1/fs
            times = np.linspace(-test_dur, 0, segment.shape[0])
            temp_data = np.zeros(test_length)
            
            sumOfInterp = 0
            sumOfProd = 0

            # tests against every datapoint in window
            for k in range(test_length):
                start = time.time_ns()
                interp_data = np.interp(model_times, times+(dt*k), proc_data)
                interp_done = time.time_ns()
                if acceleration == False:
                    inner_product = np.sum(interp_data * proc_model)
                else:
                    inner_product = inner_prod(interp_data, proc_model)
                prod_done = time.time_ns()

                sumOfInterp += interp_done-start
                sumOfProd += prod_done-interp_done

                temp_data[k] = inner_product
                if inner_product > 300:
                    events.append(((j*test_dur + k*dt), np.max(temp_data)))
                    plots.append(proc_data[k-tmpl_split:k+tmpl_split])
                    processed.append(
                        (np.linspace(
                                    (j*test_dur),
                                    (j*test_dur + k*test_length),
                                    test_length,
                                    endpoint=True
                                    ),
                         temp_data))
            if j % 10 == 0:
                print(sumOfInterp)
                print(sumOfProd)
        template_events["Template "+str(i)] = events
        template_plots["Template "+str(i)] = (proc_model, plots)

    # returns the timestamp of all events relative to the start of the data
    return template_events, template_plots, processed


# %%
# if __name__ == "__main__":
EventObject = LIGOEvent(200, 7)
strain = EventObject.get_data()
t0, dt = EventObject.get_time_vars()
fs = 1/dt.value

# %%
Template_Manager = Templates("templates/Event7TemplatesCutDown.json")
print(Template_Manager.template_count)
template = Template_Manager.get_template(0)
# %%
event_times, event_plots, ip_vals = detect_events(strain, Template_Manager, fs, tukey, 4, 0.5, acceleration=True)


# %%
def find_best_data(event_times, event_plots):
    max_values = []
    max_pos = []
    occ_times = []
    for key, event in event_times.items():
        max_values.append(np.max(event, axis=0)[1])
        max_idx = np.argmax(event, axis=0)[1]
        max_pos.append(max_idx)
        occ_times.append(event[max_idx][0])
    best_fit = np.argmax(max_values)
    print(f"The template that best fits the dataset is Template {best_fit}")
    print(f"The event occurs {occ_times[best_fit]:.7f} seconds into the dataset")
    print(f"The best fit of the event has an inner product of {max_values[best_fit]:.7f}.")

    index = max_pos[best_fit]
    key = "Template "+str(best_fit)
    model = event_plots[key][0]
    data = event_plots[key][1][index]
    plt.figure(figsize=(24, 10))
    plt.plot(model)
    plt.plot(data)


# %%
find_best_data(event_times, event_plots)

# %%
print(event_times)
# %%
template_count = Template_Manager.template_count
for i in range(template_count):
    print(Template_Manager.get_template(i).times)
# %%

print(cuda.is_available())

# %%
