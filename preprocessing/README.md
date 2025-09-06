# Physiological Signal Preprocessing (EDA & IBI)

This document contains scripts for preprocessing Electrodermal Activity (EDA) and Inter-Beat Interval (IBI) data from the WESAD dataset. The scripts extract features and generate visualizations for stress analysis, with a focus on comparing baseline (label=1) and stress (label=2) conditions.

## Scripts 

### 1. `eda_features.py`

Processes Electrodermal Activity (EDA) signals from the WESAD dataset.

#### Key Features:
- Extracts segments for baseline (label=1) and stress (label=2) from EDA signals.
- Applies windowing with different window sizes [30,45,60,75,90,120]; non overlapping.
  - EDA sampling rate: 4Hz
- Computes two types of features:
  - **Manual Features**: Basic statistical measures (mean, std, min, max, etc.)
  - **NeuroKit2 Features**: Advanced EDA metrics (phasic/tonic components, SCR features)
- Generates comprehensive visualizations:
  - Raw and Cleaned EDA signal
  - Skin Conductance Response (SCR - phasic component)
  - Skin Conductance Level (SCL - tonic component)
- Handles bad subjects (S1, S12) as noted in WESAD documentation
- Saves features to `data/eda_features_{window_size}.csv`.

#### Output:
- CSV file with extracted EDA features
- Visualization plots in `plots/EDA/` directory (baseline vs. stress comparisons).

### 2. `ibi_features.py`

Processes Blood Volume Pulse (BVP) signals to extract Inter-Beat Intervals (IBI).

#### Key Features:
- Dynamic Peak Detection: Uses an adaptive threshold approach to detect peaks in BVP signal:
  ```
     baseline = np.median(bvp)
     threshold_value = baseline + threshold * (np.percentile(bvp, 75) - baseline)
  ```
  This method adapts to individual signal characteristics and reduces noise sensitivity
- Detects peaks in BVP signal to calculate IBI
  - BVP sampling rate: 64Hz → IBI derived from peak detection
- Applies windowing with different window sizes [30,45,60,75,90,120]; non overlapping.
- Computes statistical features of IBI:
  - Mean, standard deviation, min/max, range, median
- Generates visualizations:
  - Raw BVP signal
  - Detected peaks in BVP
  - Extracted IBI series
- Saves individual IBI series to CSV files
- Handles bad subjects (S1, S12)
- Saves features to `data/ibi_features_{window_size}.csv`.

#### Output:
- CSV file with extracted IBI features
- Visualization plots in `plots/IBI/` directory (baseline vs. stress comparisons).

### Usage:
1. Place the WESAD dataset in the `Dataset/WESAD/` directory.
2. Run the scripts:
   ```
   python eda_features.py
   python ibi_features.py
   ```

Next Step: [Stress Classification]()



