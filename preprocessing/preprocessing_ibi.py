import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Feature Extraction Functions 

def extract_ibi_from_bvp_manual(bvp, fs=64): # Extracting IBI from BVP
    bvp = bvp.squeeze()
    peaks, _ = find_peaks(bvp, distance=int(fs * 0.3), prominence=0.5)
    peak_times = peaks / fs
    ibi = np.diff(peak_times)
    return ibi, peaks

def extract_ibi_features(ibi): # Manual features
    return {
        'ibi_mean': np.mean(ibi),
        'ibi_std': np.std(ibi),
        'ibi_min': np.min(ibi),
        'ibi_max': np.max(ibi),
        'ibi_range': np.max(ibi) - np.min(ibi),
        'ibi_median': np.median(ibi)
    }

# Visualization 

def visualize_ibi_manual(subject, bvp, manual_peaks, ibi_manual, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(script_dir, "plots", "ibi")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    
    # Raw BVP
    plt.subplot(3, 1, 1)
    plt.plot(bvp, color='gray')
    plt.title(f"{subject} - Raw BVP Signal")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    # peaks
    plt.subplot(3, 1, 2)
    plt.plot(bvp, label='BVP')
    plt.plot(manual_peaks, bvp[manual_peaks], 'rx', label='Peaks')
    plt.title("Peak Detection")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()

    # IBI 
    plt.subplot(3, 1, 3)
    plt.plot(ibi_manual * 1000, label='IBI (ms)', color='green')
    plt.title("Inter-Beat Intervals")
    plt.xlabel("Beat Number")
    plt.ylabel("IBI (ms)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{subject}_ibi.png"))
    plt.close()

def process_wesad_ibi(wesad_root):
    bad_subjects = ['S1', 'S12'] # bad files according to WESAD_readME
    ibi_feature_list = []
    
    ibi_output_dir = os.path.join(script_dir, "data", "ibi_csvs")
    os.makedirs(ibi_output_dir, exist_ok=True)

    for subject_folder in sorted(os.listdir(wesad_root)):
        if not subject_folder.startswith("S") or subject_folder in bad_subjects:
            continue

        subject_path = os.path.join(wesad_root, subject_folder, f"{subject_folder}.pkl")
        if not os.path.isfile(subject_path):
            print(f"Missing file for {subject_folder}")
            continue

        print(f"Processing {subject_folder}")

        try:
            with open(subject_path, 'rb') as file:
                data = pickle.load(file, encoding='latin1')

            bvp_signal = data['signal']['wrist']['BVP']
            original_labels = data['label']

            bvp_labels = np.interp(np.arange(len(bvp_signal)),
                                   np.linspace(0, len(bvp_signal) - 1, len(original_labels)),
                                   original_labels).round().astype(int)

            bvp_stress = bvp_signal[bvp_labels == 2].squeeze() # stress labelled as 2 in WESAD Dataset
            if len(bvp_stress) < 10:
                print(f"{subject_folder} — insufficient stress data")
                continue

            ibi_manual, manual_peaks = extract_ibi_from_bvp_manual(bvp_stress)
            if len(ibi_manual) < 2:
                print(f"{subject_folder} — insufficient IBI")
                continue

            ibi_feats = extract_ibi_features(ibi_manual)
            ibi_feats['subject'] = subject_folder

            ibi_csv_path = os.path.join(ibi_output_dir, f"{subject_folder}_stress_ibi.csv")
            np.savetxt(ibi_csv_path, ibi_manual, delimiter=",")

            visualize_ibi_manual(subject_folder, bvp_stress, manual_peaks, ibi_manual)

            ibi_feature_list.append(ibi_feats)
            #print(f"{subject_folder} — IBI beats: {len(ibi_manual)}")

        except Exception as e:
            print(f"Failed {subject_folder} — {e}")

    return ibi_feature_list

script_dir = os.path.dirname(os.path.abspath(__file__)) 

if __name__ == "__main__":
    wesad_path = os.path.join(script_dir, "..", "Dataset", "WESAD")
    ibi_features = process_wesad_ibi(wesad_path)
    output_csv_path = os.path.join(script_dir, "data", "ibi_features.csv")
    df_ibi = pd.DataFrame(ibi_features)
    df_ibi.to_csv(output_csv_path, index=False)
    print(f"IBI features saved to ibi_features.csv")