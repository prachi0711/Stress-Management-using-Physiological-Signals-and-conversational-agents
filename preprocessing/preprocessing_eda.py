import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk 

# Feature Extraction Functions 

def extract_stress_segment(signal, labels, stress_label=2):
    return signal[labels == stress_label] # stress labelled as 2 in WESAD Dataset

def extract_manual_eda_features(eda): # Manual feature
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

def extract_neurokit_eda_features(eda, sampling_rate=4):  # Features from Neurokit2
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

def visualize_eda(subject, eda, sampling_rate=4, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "eda")

    os.makedirs(output_dir, exist_ok=True)

    try:
        eda_cleaned = nk.eda_clean(eda, sampling_rate=sampling_rate)
        signals, info = nk.eda_process(eda_cleaned, sampling_rate=sampling_rate)

        plt.figure(figsize=(14, 10))

        # Raw EDA
        plt.subplot(4, 1, 1)
        plt.plot(eda, color='gray')
        plt.title("Raw EDA")
        plt.xlabel("Sample")
        plt.ylabel("EDA (μS)")

        # Cleaned EDA
        plt.subplot(4, 1, 2)
        plt.plot(signals["EDA_Clean"], color='blue')
        plt.title("Cleaned EDA")
        plt.xlabel("Sample")
        plt.ylabel("EDA (μS)")

        # SCR (Phasic)
        plt.subplot(4, 1, 3)
        plt.plot(signals["EDA_Phasic"], color='orange')
        plt.title("SCR (Phasic Component)")
        plt.xlabel("Sample")
        plt.ylabel("Phasic (μS)")

        # SCL (Tonic)
        plt.subplot(4, 1, 4)
        plt.plot(signals["EDA_Tonic"], color='green')
        plt.title("SCL (Tonic Component)")
        plt.xlabel("Sample")
        plt.ylabel("Tonic (μS)")

        plt.suptitle(f"{subject} - EDA Signal Breakdown", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        plot_eda = os.path.join(output_dir, f"eda_plot_{subject}.png")
        plt.savefig(plot_eda)
        plt.close()
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
            print(f"Missing file for {subject_folder}")
            continue

        print(f"Processing EDA for {subject_folder}")

        try:
            with open(subject_path, 'rb') as file:
                data = pickle.load(file, encoding='latin1')

            eda_signal = data['signal']['wrist']['EDA']
            original_labels = data['label']
            eda_labels = downsample_labels(original_labels, len(eda_signal))
            eda_stress = extract_stress_segment(eda_signal, eda_labels)

            if len(eda_stress) < 10:
                print(f"{subject_folder} — insufficient stress EDA data")
                continue

            manual_feats = extract_manual_eda_features(eda_stress)
            neurokit_feats = extract_neurokit_eda_features(eda_stress, sampling_rate=sampling_rate)

            combined_feats = {**manual_feats, **neurokit_feats}  
            combined_feats['subject'] = subject_folder
            eda_feature_list.append(combined_feats)

            visualize_eda(subject_folder, eda_stress)

            #print(f"{subject_folder} — EDA samples: {len(eda_stress)}")

        except Exception as e:
            print(f"Failed EDA for {subject_folder} — {e}")

    return eda_feature_list

# === Run ===

script_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    wesad_path = os.path.join(script_dir, "..", "Dataset", "WESAD")
    data_dir = os.path.join(script_dir, "data")
    plots_dir = os.path.join(script_dir, "plots", "eda")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    eda_features = process_eda(wesad_path)
    df_eda = pd.DataFrame(eda_features)
    df_eda.to_csv(os.path.join(data_dir, "eda_features.csv"), index=False)
    print("EDA features saved to eda_features.csv")
