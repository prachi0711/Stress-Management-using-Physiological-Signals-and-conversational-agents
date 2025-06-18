## **Dataset: WESAD (Wearable Stress and Affect Detection)** 
This dataset is a publicly available dataset, which can be downloaded from here [WESAD Dataset](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html).

### **Key Features**  
- **Modalities**: ECG, EDA (GSR), BVP, ACC, RESP, EMG (sampled at 4–64 Hz).  
- **Subjects**: 15 usable subjects (2 discarded) with lab-induced stress.  
- **Study Environment**: Laboratory.
- **Ground Truth Labels:** 1 = Baseline (neutral state); 2 = Stress (induced stress); 3 = Amusement (positive affect).

For this Project, EDA and BVP signals are used which are collected from E4 wearable sensor.

| Signal | Description | Sampling Rate | Sensor | Relevance to Stress |  
|--------|-------------|--------------|--------|---------------------|  
| **EDA** (Electrodermal Activity) | Measures skin conductance (sweat gland activity). | **4 Hz** | Empatica E4 wristband | Direct indicator of sympathetic nervous system arousal (higher EDA = stress). |  
| **BVP** (Blood Volume Pulse) | Tracks blood flow changes (used to derive HR/HRV). | **64 Hz** | Empatica E4 wristband | Heart rate variability (HRV) decreases under stress. |  


### **Stress Level Segmentation**  
To deal with stress data, Label 2 is extracted from the dataset and is then further used for preprocesssing.

- Label 2 (Stress) is further split into **low/medium/high** stress levels based on:  
  - **EDA peaks** (top 33% = high stress).  
  - **BVP-derived Interbeat Intervals**.  

### **Validation Approach**  
- **Leave-One-Subject-Out (LOSO)**: Model trained on 14 subjects, tested on the held-out subject.  

### **Ground Truth**  
- **Self-reports**: Validated via STAI (anxiety) and SSSQ (stress) questionnaires.

# Important Note

This repository does not contain the WESAD dataset files. The dataset must be downloaded separately from the official source.
Extract the contents to:

`Dataset/WESAD`

For full details, refer [WESAD dataset paper](https://doi.org/10.1145/3242969.3242985)

Next Step: [Preprocessing](https://github.com/prachi0711/Stress-Management-using-Physiological-Signals/blob/main/preprocessing/README.md)

