import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk

# feature extraction
def extract_label_segment(signal, labels, target_label):
    return signal[labels == target_label]

def extract_manual_eda_features(eda):
    eda = eda.squeeze()
    return {
        'eda_mean': np.mean(eda),
        'eda_std': np.std(eda),
        'eda_min': np.min(eda),
        'eda_max': np.max(eda),
        'eda_range': np.max(eda) - np.min(eda),
        'eda_skewness': pd.Series(eda).skew(),
        'eda_kurtosis': pd.Series(eda).kurtosis()
    }

def extract_neurokit_eda_features(eda, sampling_rate=4): # features from neurokit2
    try:
        eda_cleaned = nk.eda_clean(eda, sampling_rate=sampling_rate)
        signals, info = nk.eda_process(eda_cleaned, sampling_rate=sampling_rate)
        features = nk.eda_intervalrelated(signals, sampling_rate=sampling_rate)
        return features.iloc[0].to_dict()
    except Exception as e:
        print(f"NeuroKit EDA extraction failed: {e}")
        return {}

def downsample_labels(original_labels, target_length):
    indices = np.linspace(0, len(original_labels) - 1, target_length).astype(int)
    return original_labels[indices]

def segment_windows(signal, window_size, step=None):
    if step is None:
        step = window_size  # no overlap used but can be done by providing step size
    windows = []
    for start in range(0, len(signal) - window_size + 1, step):
        end = start + window_size
        windows.append(signal[start:end])
    return windows

# visualization
def visualize_eda(subject, eda_baseline, eda_stress, sampling_rate=4, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(script_dir, "plots", "EDA")
    os.makedirs(output_dir, exist_ok=True)

    eda_cleaned_baseline = nk.eda_clean(eda_baseline, sampling_rate=sampling_rate)
    signals_baseline, _ = nk.eda_process(eda_cleaned_baseline, sampling_rate=sampling_rate)

    eda_cleaned_stress = nk.eda_clean(eda_stress, sampling_rate=sampling_rate)
    signals_stress, _ = nk.eda_process(eda_cleaned_stress, sampling_rate=sampling_rate)

    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    fig.suptitle(f"{subject} - Baseline vs Stress EDA", fontsize=18)

    # Raw + Cleaned EDA - Baseline
    axes[0,0].plot(eda_baseline, color='red', alpha=0.7, label='Raw EDA')
    axes[0,0].plot(signals_baseline["EDA_Clean"], color='blue', label='Cleaned EDA')
    axes[0,0].set_title("Baseline - Raw & Cleaned EDA")
    axes[0,0].set_xlabel("Samples")
    axes[0,0].set_ylabel("Amplitude (μS)")
    axes[0,0].legend()

    # Raw + Cleaned EDA - Stress
    axes[0,1].plot(eda_stress, color='red', alpha=0.7, label='Raw EDA')
    axes[0,1].plot(signals_stress["EDA_Clean"], color='blue', label='Cleaned EDA')
    axes[0,1].set_title("Stress - Raw & Cleaned EDA")
    axes[0,1].set_xlabel("Samples")
    axes[0,1].set_ylabel("Amplitude (μS)")
    axes[0,1].legend()

    # Phasic Component (SCR) - Baseline
    axes[1,0].plot(signals_baseline["EDA_Phasic"], color='orange')
    axes[1,0].set_title("Baseline - Phasic (SCR)")
    axes[1,0].set_xlabel("Samples")
    axes[1,0].set_ylabel("Amplitude (μS)")

    # Phasic Component (SCR) - Stress
    axes[1,1].plot(signals_stress["EDA_Phasic"], color='orange')
    axes[1,1].set_title("Stress - Phasic (SCR)")
    axes[1,1].set_xlabel("Samples")
    axes[1,1].set_ylabel("Amplitude (μS)")

    # Tonic Component (SCL) - Baseline
    axes[2,0].plot(signals_baseline["EDA_Tonic"], color='green')
    axes[2,0].set_title("Baseline - Tonic (SCL)")
    axes[2,0].set_xlabel("Samples")
    axes[2,0].set_ylabel("Amplitude (μS)")

    # Tonic Component (SCL) - Stress
    axes[2,1].plot(signals_stress["EDA_Tonic"], color='green')
    axes[2,1].set_title("Stress - Tonic (SCL)")
    axes[2,1].set_xlabel("Samples")
    axes[2,1].set_ylabel("Amplitude (μS)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path = os.path.join(output_dir, f"{subject}_eda_plot.png")
    plt.savefig(plot_path)
    plt.close()

def process_eda(wesad_root):
    bad_subjects = ['S1', 'S12']
    sampling_rate = 4
    
    # different window sizes in seconds and steps (set steps for overlapping)
    window_configs = {
        30: {'size': 30 * sampling_rate, 'step': 30 * sampling_rate},
        45: {'size': 45 * sampling_rate, 'step': 45 * sampling_rate},
        60: {'size': 60 * sampling_rate, 'step': 60 * sampling_rate},
        75: {'size': 75 * sampling_rate, 'step': 75 * sampling_rate},
        90: {'size': 90 * sampling_rate, 'step': 90 * sampling_rate},
        120: {'size': 120 * sampling_rate, 'step': 120 * sampling_rate}
    }
    
    dfs = {size: [] for size in window_configs}

    for subject_folder in sorted(os.listdir(wesad_root)):
        if not subject_folder.startswith("S") or subject_folder in bad_subjects:
            continue

        subject_path = os.path.join(wesad_root, subject_folder, f"{subject_folder}.pkl")
        if not os.path.isfile(subject_path):
            continue

        #print(f"Processing EDA for {subject_folder}")

        try:
            with open(subject_path, 'rb') as file:
                data = pickle.load(file, encoding='latin1')

            eda_signal = data['signal']['wrist']['EDA']
            original_labels = data['label']
            eda_labels = downsample_labels(original_labels, len(eda_signal))

            for label_value, label in [(1, "baseline"), (2, "stress")]:
                eda_segment = extract_label_segment(eda_signal, eda_labels, label_value)

                for window_sec, config in window_configs.items():
                    windows = segment_windows(
                        eda_segment, 
                        window_size=config['size'], 
                        step=config['step']
                    )

                    for i, window in enumerate(windows):
                        if len(window) < 10:
                            continue

                        feats_manual = extract_manual_eda_features(window)
                        feats_nk = extract_neurokit_eda_features(window, sampling_rate=sampling_rate)
                        all_feats = {**feats_manual, **feats_nk}
                        all_feats["subject"] = subject_folder
                        all_feats["label"] = label
                        all_feats["window"] = i+1
                        dfs[window_sec].append(all_feats)

            visualize_eda(subject_folder,
                          extract_label_segment(eda_signal, eda_labels, 1),
                          extract_label_segment(eda_signal, eda_labels, 2))

        except Exception as e:
            print(f"Failed EDA for {subject_folder}: {e}")

    return {size: pd.DataFrame(feats) for size, feats in dfs.items()}

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wesad_path = os.path.join(script_dir, "..", "..", "Dataset", "WESAD")
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    eda_dfs = process_eda(wesad_path)

    for window_sec, df_eda in eda_dfs.items():
        # dropping empty columns as they are not available for 30s window size
        df_eda = df_eda.drop(
            columns=[col for col in ["EDA_Sympathetic", "EDA_SympatheticN", "EDA_Autocorrelation"] 
                    if col in df_eda.columns]
        )
        csv_path = os.path.join(data_dir, f"eda_features_{window_sec}.csv")
        df_eda.to_csv(csv_path, index=False)