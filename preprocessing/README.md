# Physiological Signal Preprocessing (EDA & IBI)

This document contains scripts for preprocessing Electrodermal Activity (EDA) and Inter-Beat Interval (IBI) data from the WESAD dataset. The scripts extract features and generate visualizations for stress analysis, with a focus on comparing baseline (label=1) and stress (label=2) conditions.

## Scripts 

### 1. `preprocessing_eda.py`

Processes Electrodermal Activity (EDA) signals from the WESAD dataset.

#### Key Features:
- Extracts segments for baseline (label=1) and stress (label=2) from EDA signals.
- Computes two types of features:
  - **Manual Features**: Basic statistical measures (mean, std, min, max, etc.)
  - **NeuroKit2 Features**: Advanced EDA metrics (phasic/tonic components, SCR features)
- Generates comprehensive visualizations:
  - Raw EDA signal
  - Cleaned EDA signal
  - Skin Conductance Response (SCR - phasic component)
  - Skin Conductance Level (SCL - tonic component)
- Handles bad subjects (S1, S12) as noted in WESAD documentation
- Saves features to `data/eda_features_label.csv`.

#### Output:
- CSV file with extracted EDA features
- Visualization plots in `plots/EDA/` directory (side-by-side baseline vs. stress plots)

### 2. `preprocessing_ibi.py`

Processes Blood Volume Pulse (BVP) signals to extract Inter-Beat Intervals (IBI).

#### Key Features:
- Detects peaks in BVP signal to calculate IBI
- Computes statistical features of IBI:
  - Mean, standard deviation, min/max, range, median
- Generates visualizations:
  - Raw BVP signal
  - Detected peaks in BVP
  - Extracted IBI series
- Saves individual IBI series to CSV files
- Handles bad subjects (S1, S12)
- Saves features to `data/ibi_features_label.csv`

#### Output:
- CSV file with extracted IBI features
- Individual IBI series in `data/ibi_csvs/`
- Visualization plots in `plots/IBI/` directory (baseline vs. stress comparisons).

### Usage:
1. Place the WESAD dataset in the `Dataset/WESAD/` directory.
2. Run the scripts:
   ```
   python preprocessing_eda.py
   python preprocessing_ibi.py
   ```

Next Step: [Stress Classification]()
