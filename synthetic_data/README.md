# Synthetic Data Generation

This module provides utilities to generate synthetic physiological signals (EDA and BVP) using [NeuroKit2](https://github.com/neuropsychology/NeuroKit)

It is useful for testing and debugging the stress inference pipeline when real data is not available.

## Files

* `data_generation.py` – Script to generate synthetic EDA and BVP signals.
* Output files:

  * `synthetic_eda.csv` → Raw EDA signal
  * `synthetic_bvp.csv` → Raw BVP signal


## How it works

* EDA (Electrodermal Activity):

  * Generated at `4 Hz` (similar to Empatica E4 in WESAD dataset).
* BVP (Blood Volume Pulse):

  * Generated at `64 Hz` (similar to Empatica E4 BVP sensor).
  * Default simulated heart rate: \~70 bpm.
* Duration: **3 minutes (180 seconds)**

Both signals are saved as `.csv` files in the working directory.

## Usage

Run the script:

```bash
python data_generation.py
```

## Next Steps

1. **Preprocessing & Feature Extraction**

   * Once the raw synthetic signals are generated, run the preprocessing and feature extraction scripts provided in the **`preprocessing/`** folder.
   * This step will clean the signals and extract features required for stress classification.

2. **Integration with ROS2 Nodes**

   * Save the preprocessed feature CSVs inside the ROS2 node folder (e.g., `stress_inference_pkg/`) so that they can be used for testing inference nodes (`stress_inference_node.py`, `stress_inference_rf_node.py`).


## Requirements

Install dependencies:

```bash
pip install neurokit2 pandas
```


## **Important Note**

* The generated signals are synthetic and do not represent real physiological data.
* They are intended for testing the pipeline (preprocessing, feature extraction, ROS nodes), not for actual stress inference.

