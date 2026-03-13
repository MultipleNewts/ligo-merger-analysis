# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from analysis_techniques.welch_method import welch
from analysis_techniques.data_processing import whiten, bandpass


# %%
# processes data (whitens and bandpasses)
def process_data(data, fs, freqs, PSD):
    whitened_data = whiten(data, fs, freqs, PSD, tukey)
    bp_data = bandpass(whitened_data, fs, order=8)
    return bp_data


# %%
def detect_events_v2(full_data, template_bank, fs=4096, window_func=tukey, seg_dur=4, overlap=0.75):

    # Compute PSD
    freqs, PSD = welch(full_data, fs, window_func, int(seg_dur*fs), overlap=overlap)
    dt = 1/fs
    test_length_indexes = int(4*fs)
    if test_length_indexes % 2 != 0:
        test_length_indexes -= 1

    # Moving window setup
    overlap = 0.5
    step = int(test_length_indexes*(1-overlap))
    segments = []

    template_events = {}
    template_plots = {}
    template_inner_products = {}

    max_iter = int((len(full_data) - test_length_indexes) // step) + 1
    for i in range(max_iter):
        index = int(i*step)
        segments.append(full_data[index:(index+test_length_indexes)])

    for t in range(template_bank.template_count):
        print(f"Reached template {t}")

        current_template = template_bank.get_template(t)
        print(f"Masses {current_template.mass1}, {current_template.mass2}")

        current_model = current_template.hp[:]

        processed_model = process_data(current_model, fs, freqs, PSD)
        current_model_times = current_template.times[:]
        current_model_t0 = - current_model_times[0]
        events = []
        plots = []
        inner_products_with_time = []

        for j, segment in enumerate(segments):
            processed_segment = process_data(segment, fs, freqs, PSD)
            indexes_to_convolve = processed_segment.shape[0] - processed_model.shape[0]

            inner_products = np.zeros(indexes_to_convolve)
            detection = False

            for i in range(indexes_to_convolve):
                inner_product = np.sum(processed_segment[i:processed_model.shape[0] + i] * processed_model)
                inner_products[i] = inner_product

                if inner_product > 300:
                    detection = True
                    events.append(((j*step*dt + i*dt + current_model_t0), inner_product))
                    starting_t0 = j*step*dt + i*dt
                    ending_t0 = j*step*dt + (processed_model.shape[0]+i)*dt
                    current_times = np.linspace(starting_t0, ending_t0, processed_model.shape[0], endpoint=False)
                    plots.append((processed_segment[i:processed_model.shape[0] + i], processed_model, current_times))

            if detection:
                inner_products_with_time.append((
                    np.linspace(
                        (j*step*dt + current_model_t0),
                        (j*step*dt + indexes_to_convolve*dt + current_model_t0),
                        indexes_to_convolve,
                        endpoint=False
                    ),
                    inner_products
                ))
        template_events["Template " + str(t)] = events
        template_plots["Template "+str(t)] = plots
        template_inner_products["Template " + str(t)] = inner_products_with_time
    return template_events, template_plots, template_inner_products


# %%
# %%
def find_best_data_v2(template_bank, event_times, event_plots):

    max_values = []
    max_values_index = []
    occured_times = []
    templates_used = []

    for key, event in event_times.items():
        max_values.append(np.max(event, axis=0)[1])
        max_index = np.argmax(event, axis=0)[1]
        max_values_index.append(max_index)

        occured_times.append(event[max_index][0])
        template_used = int(key[9:])
        templates_used.append(template_used)

    best_fit = np.argmax(max_values)
    best_template = templates_used[best_fit]
    occured_time = occured_times[best_fit]

    print(f"The template that best fits the dataset is Template {best_template}")
    print(f"The event occurs {occured_time:.7f} seconds into the dataset")
    print(f"The best fit of the event has an inner product of {max_values[best_fit]:.7f}.")

    best_model = template_bank.get_template(best_template)
    M1, M2 = (best_model.mass1, best_model.mass2)
    print(f"This suggests the black hole masses involved were M1: {M1} & M2: {M2} in solar masses.")

    key = "Template " + str(best_template)
    index = max_values_index[best_fit]
    model = event_plots[key][index][1]
    data = event_plots[key][index][0]
    times = event_plots[key][index][2]

    solM = r"$M_{\odot}$"
    plt.figure(figsize=(24, 10))
    plt.plot(times, model, label=f"Model: {M1}{solM}; {M2}{solM}")
    plt.plot(times, data, label="Data")
    plt.title("Best Fitting Model for Detected BH-BH Merger Event", fontsize=30)
    plt.xlabel(r"Times ($s$)", fontsize=25)
    plt.xticks(fontsize=20)
    plt.ylabel(r"Standard Deviations ($\sigma$)", fontsize=25)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=25)
    plt.show()

    return (times, data, model)

# %%
