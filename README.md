# LIGO Blackhole Merger Event Analysis
Project focused on locating blackhole merger events in LIGO data. Focuses on data analysis and low SNR techniques.

The main pipeline can be found in the folder `analysis_techniques/final_detector_v2.py` and the associated modules can be found in the same folder.

# Roadmap
- [x] Developing Models
    - [x] Handling Noise
    - [x] Understanding Signals
- [x] Analysing Real Data
    - [x] Locating Signal of Known Timestamp
    - [x] Locating a Signal within a Window
- [x] Locate Signal(s) from Unseen Dataset
- [x] Analyse Characteristics of Merger Signals

# Goals
The primary end goal of this project is to understand the various methods used in extracting signals from low SNR data and attmepting to classify these results. Whilst a number of libraries are available for this exact purpose, we wish to study and develop various algorithms ourselves to better understand the processes being used. An optimal end goal would be to classify various properties of all events detected in a large time window, using primarily our own algorithms.

# Contribution
The primary contributor for each major algorithm was as follows:
- Welch PSD Computation: Matthew Seldon
- Data Whitening: Benjamin Wigley
- Data Search: Matthew Seldon
- Template Matching: Benjamin Wigley

# Results
The final processed data for event GW240104_16493 can be seen matched to a template below.
![Matched Signal](https://github.com/MultipleNewts/ligo-merger-analysis/blob/main/Final_Images/BestFitFinal.png)

Additionally, a heat map of inner-products (representing the quality of a fit) can be seen here:
![Inner-Product Heat Map](https://github.com/MultipleNewts/ligo-merger-analysis/blob/main/Final_Images/MassHeatmap.png)

## Abstract from Final Report
This project implemented a computational pipeline for use in the detection and analysis of black hole
merger events. This included power spectral density estimation for non-stationary signals, data whiten-
ing, and waveform template matching. The algorithms were compared with their counterparts from
established scientific libraries. The analysis pipeline identified a possible event with component masses
of $47^{+13}_{−8}M_{\odot}$ and $47^{+9}_{−15}M_{\odot}$. Within the analysed events, no clear correlation between signal-to-noise ratios and component masses was observed. Additionally, template matching caused bands of increased
probability to occur in a semi-periodic pattern around the best-fit template; these likely arose due to
similarities in waveform structure at these periodic intervals.
