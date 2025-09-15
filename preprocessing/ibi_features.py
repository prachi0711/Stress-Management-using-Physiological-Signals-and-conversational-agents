import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# feature extraction
def extract_ibi_from_bvp_manual(bvp, fs=64, threshold=5):  
    bvp = bvp.squeeze()
    # dynamic threshold as signals are different per subject (personalized)
    baseline = np.median(bvp)
    threshold_value = baseline + threshold * (np.percentile(bvp, 75) - baseline)
    
    peaks, _ = find_peaks(bvp, 
                         distance=int(fs * 0.3),  
                         height=threshold_value,   
                         prominence=(0.5, None))  
    peak_times = peaks / fs
    ibi = np.diff(peak_times)
    return ibi, peaks, threshold_value

def extract_ibi_features(ibi):
    if len(ibi) < 2: 
        return None
    return {
        'ibi_mean': np.mean(ibi),
        'ibi_std': np.std(ibi),
        'ibi_min': np.min(ibi),
        'ibi_max': np.max(ibi),
        'ibi_range': np.max(ibi) - np.min(ibi),
        'ibi_median': np.median(ibi),
    }

def segment_windows(signal, window_size, step=None):
    if step is None:
        step = window_size # no overlap used but can be done by providing step size
    windows = []
    for start in range(0, len(signal) - window_size + 1, step):
        end = start + window_size
        windows.append(signal[start:end])
    return windows

# visualization
def visualize_ibi(subject, bvp_baseline, peaks_baseline, ibi_baseline, threshold_baseline,
                  bvp_stress, peaks_stress, ibi_stress, threshold_stress, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(script_dir, "plots", "IBI")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f"{subject} - Baseline vs Stress IBI Analysis", fontsize=16)

    # Raw BVP - Baseline
    axes[0,0].plot(bvp_baseline)
    axes[0,0].axhline(y=threshold_baseline, color='r', linestyle='--', label='Threshold')
    axes[0,0].set_title("Baseline - Raw BVP with Threshold")
    axes[0,0].set_xlabel("Samples")
    axes[0,0].set_ylabel("Amplitude (nW)")
    axes[0,0].legend()

    # Raw BVP - Stress
    axes[0,1].plot(bvp_stress)
    axes[0,1].axhline(y=threshold_stress, color='r', linestyle='--', label='Threshold')
    axes[0,1].set_title("Stress - Raw BVP with Threshold")
    axes[0,1].set_xlabel("Samples")
    axes[0,1].set_ylabel("Amplitude (nW)")
    axes[0,1].legend()

    # Peak detection - Baseline
    axes[1,0].plot(bvp_baseline, label="BVP")
    axes[1,0].plot(peaks_baseline, bvp_baseline[peaks_baseline], 'rx', label="Detected Peaks")
    axes[1,0].set_title("Baseline - Peak Detection")
    axes[1,0].set_xlabel("Samples")
    axes[1,0].set_ylabel("Amplitude (nW)")
    axes[1,0].legend()

    # Peak detection - Stress
    axes[1,1].plot(bvp_stress, label="BVP")
    axes[1,1].plot(peaks_stress, bvp_stress[peaks_stress], 'rx', label="Detected Peaks")
    axes[1,1].set_title("Stress - Peak Detection")
    axes[1,1].set_xlabel("Samples")
    axes[1,1].set_ylabel("Amplitude (nW)")
    axes[1,1].legend()

    # IBI series - Baseline
    axes[2,0].plot(ibi_baseline * 1000, 'bo-')
    axes[2,0].set_title("Baseline - IBI Series")
    axes[2,0].set_xlabel("Beat Number")
    axes[2,0].set_ylabel("IBI (ms)")
    axes[2,0].grid(True)

    # IBI series - Stress
    axes[2,1].plot(ibi_stress * 1000, 'ro-')
    axes[2,1].set_title("Stress - IBI Series")
    axes[2,1].set_xlabel("Beat Number")
    axes[2,1].set_ylabel("IBI (ms)")
    axes[2,1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{subject}_ibi_plot.png"))
    plt.close()
    
def process_ibi(wesad_root):
    bad_subjects = ['S1', 'S12']
    fs = 64
    
    # different window sizes in seconds and steps (set steps for overlapping)
    window_configs = {
        30: {'size': 30 * fs, 'step': 30 * fs},
        45: {'size': 45 * fs, 'step': 45 * fs},
        60: {'size': 60 * fs, 'step': 60 * fs},
        75: {'size': 75 * fs, 'step': 75 * fs},
        90: {'size': 90 * fs, 'step': 90 * fs},
        120: {'size': 120 * fs, 'step': 120 * fs}
    }
    
    dfs = {size: [] for size in window_configs}

    for subject_folder in sorted(os.listdir(wesad_root)):
        if not subject_folder.startswith("S") or subject_folder in bad_subjects:
            continue

        subject_path = os.path.join(wesad_root, subject_folder, f"{subject_folder}.pkl")
        if not os.path.isfile(subject_path):
            continue

        print(f"Processing IBI for {subject_folder}")

        try:
            with open(subject_path, 'rb') as file:
                data = pickle.load(file, encoding='latin1')

            bvp_signal = data['signal']['wrist']['BVP']
            labels = data['label']
            bvp_labels = np.interp(np.arange(len(bvp_signal)),
                                 np.linspace(0, len(bvp_signal)-1, len(labels)),
                                 labels).round().astype(int)

            baseline_ibi, baseline_peaks, baseline_threshold = extract_ibi_from_bvp_manual(
                bvp_signal[bvp_labels==1])
            stress_ibi, stress_peaks, stress_threshold = extract_ibi_from_bvp_manual(
                bvp_signal[bvp_labels==2])

            for label_value, label in [(1, "baseline"), (2, "stress")]:
                bvp_segment = bvp_signal[bvp_labels == label_value].squeeze()

                for window_sec, config in window_configs.items():
                    windows = segment_windows(
                        bvp_segment,
                        window_size=config['size'],
                        step=config['step']
                    )

                    for i, window in enumerate(windows):
                        ibi, peaks, _ = extract_ibi_from_bvp_manual(window)
                        if len(ibi) < 2:
                            continue

                        feats = extract_ibi_features(ibi)
                        if feats is None:
                            continue
                            
                        feats['subject'] = subject_folder
                        feats['label'] = label
                        feats['window'] = i+1
                        dfs[window_sec].append(feats)

            visualize_ibi(
                subject_folder,
                bvp_signal[bvp_labels==1],
                baseline_peaks,
                baseline_ibi,
                baseline_threshold,
                bvp_signal[bvp_labels==2],
                stress_peaks,
                stress_ibi,
                stress_threshold
            )

        except Exception as e:
            print(f"Failed IBI for {subject_folder}: {e}")

    return {size: pd.DataFrame(feats) for size, feats in dfs.items()}

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wesad_path = os.path.join(script_dir, "..", "..", "Dataset", "WESAD")
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    ibi_dfs = process_ibi(wesad_path)

    for window_sec, df_ibi in ibi_dfs.items():
        csv_path = os.path.join(data_dir, f"ibi_features_{window_sec}.csv")
        df_ibi.to_csv(csv_path, index=False)