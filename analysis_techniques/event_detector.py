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
def detect_events(full_data, template_bank, fs=4096, window_func=tukey, seg_dur=4, overlap=0.75):
    """
    Runs through a full dataset and template matches against a set of templates,
    any scans above a detection threshold are returned.

    Parameters
    ----------
    full_data : `array`
        the full dataset to be scanned
    template_bank : `Templates`
        a template bank Object
    fs : `int`
        the sampling frequency, defaults to `4096`
    window_func : `FunctionType`
        function pointer for window function, defaults to `tukey`
    seg_dur : `float`
        duration, in seconds, for Welch PSD computation
    overlap : `float`
        the overlap ratio for Welch PSD computation

    Returns
    -------
    template_events : `dict`
        dictionary containing the timestamp and the maximum inner_product for any
        events that surpass the threshold
    template_plots : `dict`
        the processed data for each event detection, matches the size of the template
    processed : `array`
        inner product data for each event detection, mostly for debugging purposes
    """
    # compute Welch PSD
    freqs, PSD = welch(full_data, fs, window_func, int(seg_dur*fs), overlap=overlap)
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
            # if j % 10 == 0:
            #     print(f"Processing Segment {j}...")
            # processes each window
            proc_data = process_data(segment, fs, freqs, PSD)
            test_dur = test_length/fs
            dt = 1/fs
            times = np.linspace(-test_dur, 0, segment.shape[0])
            temp_data = np.zeros(test_length)

            extra_times_start_point = times[-1] + dt
            extra_times_end_point = times[-1] * (dt * test_length)
            extra_times = np.arange(extra_times_start_point, extra_times_end_point, test_length)

            total_times = np.concatenate((times, extra_times))
            total_interp = np.interp(model_times, total_times, proc_data)

            # tests against every datapoint in window
            for k in range(test_length):
                interp_data = total_interp[k:k+test_length]
                # interp_data = np.interp(model_times, times+(dt*k), proc_data)
                inner_product = np.sum(interp_data * proc_model)
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
        template_events["Template "+str(i)] = events
        template_plots["Template "+str(i)] = (proc_model, plots)

    # returns the timestamp of all events relative to the start of the data
    return template_events, template_plots, processed


# %%
def find_best_data(template_bank, event_times, event_plots, fs=4096):
    """
    Takes data from `detect_events()` and finds the best fitting model for
    the detected event (if there is one)

    Parameters
    ----------
    template_bank : `Templates`
        a template bank Object
    event_times : `dict`
        dictionary containing the timestamp and the maximum inner_product for any
        events that surpass the threshold
    event_plots : `dict`
        the processed data for each event detection, matches the size of the template
    fs : `float`
        the sampling frequency of the data

    Returns
    -------
    times : `array`
        an array containing the times of each datapoint for the best-fitting model
    data : `array`
        an array containing the processed data for the best-fitting model
    model : `array`
        an array containing the model data for the best-fitting model
    """
    dt = 1/fs
    max_values = []
    max_pos = []
    occ_times = []
    for key, event in event_times.items():
        max_values.append(np.max(event, axis=0)[1])
        max_idx = np.argmax(event, axis=0)[1]
        max_pos.append(max_idx)
        occ_times.append(event[max_idx][0])
    best_fit = np.argmax(max_values)
    occurred = occ_times[best_fit]
    print(f"The template that best fits the dataset is Template {best_fit}")
    print(f"The event occurs {occ_times[best_fit]:.7f} seconds into the dataset")
    print(f"The event occurs {occurred:.7f} seconds into the dataset")
    print(f"The best fit of the event has an inner product of {max_values[best_fit]:.7f}.")

    best_model = template_bank.get_template(best_fit)
    M1, M2 = (best_model.mass1, best_model.mass2)
    print(f"This suggests the black hole masses involved were M1: {M1} & M2: {M2} in solar masses.")

    index = max_pos[best_fit]
    key = "Template "+str(best_fit)
    model = event_plots[key][0]
    data = event_plots[key][1][index]
    incr = dt*len(data)/2
    times = np.linspace(occurred-incr, occurred+incr, len(data))
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
