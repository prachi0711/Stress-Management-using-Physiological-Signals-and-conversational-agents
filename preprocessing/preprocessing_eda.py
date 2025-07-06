import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk


# Feature Extraction Functions 
def extract_label_segment(signal, labels, target_label):
    return signal[labels == target_label] # Extract stress and baseline label from WESAD dataset


def extract_manual_eda_features(eda): # Manual features
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


def extract_neurokit_eda_features(eda, sampling_rate=4): # Features from Neurokit2
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


# Visualization 

def visualize_eda(subject, eda_baseline, eda_stress, sampling_rate=4, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "EDA")

    os.makedirs(output_dir, exist_ok=True)

    try:
        eda_cleaned_baseline = nk.eda_clean(eda_baseline, sampling_rate=sampling_rate)
        signals_baseline, _ = nk.eda_process(eda_cleaned_baseline, sampling_rate=sampling_rate)

        eda_cleaned_stress = nk.eda_clean(eda_stress, sampling_rate=sampling_rate)
        signals_stress, _ = nk.eda_process(eda_cleaned_stress, sampling_rate=sampling_rate)

        fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex='row')
        fig.suptitle(f"{subject} - Baseline vs Stress EDA", fontsize=18)

        # Raw EDA
        axes[0,0].plot(eda_baseline, color='gray')
        axes[0,0].set_title("Baseline - Raw EDA")
        axes[0,0].set_ylabel("μS")

        axes[0,1].plot(eda_stress, color='gray')
        axes[0,1].set_title("Stress - Raw EDA")

        # Cleaned
        axes[1,0].plot(signals_baseline["EDA_Clean"], color='blue')
        axes[1,0].set_title("Baseline - Cleaned EDA")

        axes[1,1].plot(signals_stress["EDA_Clean"], color='blue')
        axes[1,1].set_title("Stress - Cleaned EDA")

        # Phasic
        axes[2,0].plot(signals_baseline["EDA_Phasic"], color='orange')
        axes[2,0].set_title("Baseline - Phasic (SCR)")

        axes[2,1].plot(signals_stress["EDA_Phasic"], color='orange')
        axes[2,1].set_title("Stress - Phasic (SCR)")

        # Tonic
        axes[3,0].plot(signals_baseline["EDA_Tonic"], color='green')
        axes[3,0].set_title("Baseline - Tonic (SCL)")
        axes[3,0].set_xlabel("Sample")

        axes[3,1].plot(signals_stress["EDA_Tonic"], color='green')
        axes[3,1].set_title("Stress - Tonic (SCL)")
        axes[3,1].set_xlabel("Sample")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_eda = os.path.join(output_dir, f"eda_plot_{subject}.png")
        plt.savefig(plot_eda)
        plt.close()

    except Exception as e:
        print(f"Visualization failed for {subject}: {e}")


# Main Processing 

def process_eda(wesad_root):
    bad_subjects = ['S1', 'S12'] # bad files according to WESAD_readME
    eda_feature_list = []
    sampling_rate = 4

    for subject_folder in sorted(os.listdir(wesad_root)):
        if not subject_folder.startswith("S") or subject_folder in bad_subjects:
            continue

        subject_path = os.path.join(wesad_root, subject_folder, f"{subject_folder}.pkl")
        if not os.path.isfile(subject_path):
            #print(f"Missing file for {subject_folder}")
            continue

        #print(f"Processing EDA for {subject_folder}")

        try:
            with open(subject_path, 'rb') as file:
                data = pickle.load(file, encoding='latin1')

            eda_signal = data['signal']['wrist']['EDA']
            original_labels = data['label']
            eda_labels = downsample_labels(original_labels, len(eda_signal))

            eda_baseline = extract_label_segment(eda_signal, eda_labels, target_label=1)
            eda_stress = extract_label_segment(eda_signal, eda_labels, target_label=2)

            if len(eda_baseline) < 10 or len(eda_stress) < 10:
                print(f"{subject_folder} — insufficient data for baseline or stress")
                continue

            # Baseline features
            manual_feats_baseline = extract_manual_eda_features(eda_baseline)
            neurokit_feats_baseline = extract_neurokit_eda_features(eda_baseline, sampling_rate=sampling_rate)
            combined_baseline = {**manual_feats_baseline, **neurokit_feats_baseline}
            combined_baseline['subject'] = subject_folder
            combined_baseline['label'] = "baseline"
            eda_feature_list.append(combined_baseline)

            # Stress features
            manual_feats_stress = extract_manual_eda_features(eda_stress)
            neurokit_feats_stress = extract_neurokit_eda_features(eda_stress, sampling_rate=sampling_rate)
            combined_stress = {**manual_feats_stress, **neurokit_feats_stress}
            combined_stress['subject'] = subject_folder
            combined_stress['label'] = "stress"
            eda_feature_list.append(combined_stress)

            visualize_eda(subject_folder, eda_baseline, eda_stress)

        except Exception as e:
            print(f"Failed EDA for {subject_folder} — {e}")

    return eda_feature_list


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wesad_path = os.path.join(script_dir, "..", "Dataset", "WESAD")
    data_dir = os.path.join(script_dir, "data")
    plots_dir = os.path.join(script_dir, "plots", "EDA")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    eda_features = process_eda(wesad_path)
    df_eda = pd.DataFrame(eda_features)
    df_eda.to_csv(os.path.join(data_dir, "eda_features_label.csv"), index=False)
    print("EDA features saved to eda_features_label.csv")
