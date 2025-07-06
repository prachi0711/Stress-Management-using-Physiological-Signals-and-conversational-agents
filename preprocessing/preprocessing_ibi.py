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
def visualize_ibi_baseline_stress(subject, bvp_baseline, peaks_baseline, ibi_baseline,
                                  bvp_stress, peaks_stress, ibi_stress, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(script_dir, "plots", "IBI")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Raw BVP
    axes[0, 0].plot(bvp_baseline, color='gray')
    axes[0, 0].set_title("Baseline - Raw BVP")
    axes[0, 0].set_ylabel("Amplitude")

    axes[0, 1].plot(bvp_stress, color='gray')
    axes[0, 1].set_title("Stress - Raw BVP")

    # Peaks
    axes[1, 0].plot(bvp_baseline, label="BVP")
    axes[1, 0].plot(peaks_baseline, bvp_baseline[peaks_baseline], 'rx', label="Peaks")
    axes[1, 0].set_title("Baseline - Peak Detection")
    axes[1, 0].legend()

    axes[1, 1].plot(bvp_stress, label="BVP")
    axes[1, 1].plot(peaks_stress, bvp_stress[peaks_stress], 'rx', label="Peaks")
    axes[1, 1].set_title("Stress - Peak Detection")
    axes[1, 1].legend()

    # IBI
    axes[2, 0].plot(ibi_baseline * 1000, color='blue')
    axes[2, 0].set_title("Baseline - IBI (ms)")
    axes[2, 0].set_xlabel("Beat Number")
    axes[2, 0].set_ylabel("IBI (ms)")

    axes[2, 1].plot(ibi_stress * 1000, color='red')
    axes[2, 1].set_title("Stress - IBI (ms)")
    axes[2, 1].set_xlabel("Beat Number")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path = os.path.join(output_dir, f"{subject}_ibi_plot.png")
    plt.savefig(plot_path)
    plt.close()

# Main Processing 
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

            ibi_data = {}

            for label_value, label in [(1, "baseline"), (2, "stress")]: # Extract stress and baseline label from WESAD dataset
                bvp_cond = bvp_signal[bvp_labels == label_value].squeeze()

                if len(bvp_cond) < 10:
                    #print(f"{subject_folder} — insufficient {label} data")
                    continue

                ibi_manual, peaks_manual = extract_ibi_from_bvp_manual(bvp_cond)

                if len(ibi_manual) < 2:
                    #print(f"{subject_folder} — insufficient IBI for {label}")
                    continue

                ibi_feats = extract_ibi_features(ibi_manual)
                ibi_feats['subject'] = subject_folder
                ibi_feats['label'] = label
                ibi_feature_list.append(ibi_feats)

                ibi_csv_path = os.path.join(
                    ibi_output_dir, f"{subject_folder}_{label}_ibi.csv"
                )
                np.savetxt(ibi_csv_path, ibi_manual, delimiter=",")

                ibi_data[f"{label}_bvp"] = bvp_cond
                ibi_data[f"{label}_peaks"] = peaks_manual
                ibi_data[f"{label}_ibi"] = ibi_manual

            if all(k in ibi_data for k in ["baseline_bvp", "baseline_peaks", "baseline_ibi",
                                           "stress_bvp", "stress_peaks", "stress_ibi"]):
                visualize_ibi_baseline_stress(
                    subject_folder,
                    ibi_data["baseline_bvp"],
                    ibi_data["baseline_peaks"],
                    ibi_data["baseline_ibi"],
                    ibi_data["stress_bvp"],
                    ibi_data["stress_peaks"],
                    ibi_data["stress_ibi"]
                )

        except Exception as e:
            print(f"Failed {subject_folder} — {e}")

    return ibi_feature_list

script_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    wesad_path = os.path.join(script_dir, "..", "Dataset", "WESAD")
    ibi_features = process_wesad_ibi(wesad_path)
    output_csv_path = os.path.join(script_dir, "data", "ibi_features_label.csv")
    df_ibi = pd.DataFrame(ibi_features)
    df_ibi.to_csv(output_csv_path, index=False)
    print(f"IBI features saved to ibi_features_label.csv")
