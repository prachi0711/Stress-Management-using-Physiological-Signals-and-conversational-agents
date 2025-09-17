import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import entropy
from dialogue_manager import DialogueManager  
import joblib

# LOSO evaluation
def loso_rf_with_dialogue(save_models=True):
    eda_df = pd.read_csv("preprocessing/data/eda_features_60.csv")
    ibi_df = pd.read_csv("preprocessing/data/ibi_features_60.csv")
    df = pd.merge(eda_df, ibi_df, on=["subject", "label", "window"])
    df["label_num"] = df["label"].map({"baseline": 0, "stress": 1})

    top_features = ['EDA_Tonic_SD', 'SCR_Peaks_Amplitude_Mean', 'SCR_Peaks_N', 'eda_kurtosis',
                    'eda_max', 'eda_mean', 'eda_min', 'eda_range', 'eda_skewness', 'eda_std',
                    'ibi_max', 'ibi_mean', 'ibi_median', 'ibi_min', 'ibi_range', 'ibi_std']

    subjects = df["subject"].unique()
    all_true, all_pred, all_entropy = [], [], []

    dialogue_manager = DialogueManager(entropy_threshold=0.45, use_time=True)

    for test_subj in subjects:
        train_df = df[df["subject"] != test_subj]
        test_df = df[df["subject"] == test_subj]

        X_train = train_df[top_features]
        y_train = train_df["label_num"]
        X_test = test_df[top_features]
        y_test = test_df["label_num"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)

        y_proba = rf.predict_proba(X_test_scaled)
        y_pred = np.argmax(y_proba, axis=1)
        sample_entropy = entropy(y_proba.T)

        for i in range(len(y_pred)):
            entropy_val = sample_entropy[i]
            pred_label = y_pred[i]

            timestamp = datetime.datetime.now()
            response = dialogue_manager.get_response(
                user_id=test_subj,
                entropy_val=entropy_val,
                pred_label=pred_label,
                timestamp=timestamp,
                sample_idx=None
            )
    

            all_true.append(y_test.iloc[i])
            all_pred.append(pred_label)
            all_entropy.append(entropy_val)

    # UQ results
    print("Accuracy:", round(accuracy_score(all_true, all_pred), 4))
    print(confusion_matrix(all_true, all_pred))
    print(classification_report(all_true, all_pred, target_names=["baseline", "stress"]))

    threshold = dialogue_manager.entropy_threshold
    correct_certain = correct_uncertain = 0
    incorrect_certain = incorrect_uncertain = 0

    for t, p, e in zip(all_true, all_pred, all_entropy):
        if p == t:
            if e > threshold:
                correct_uncertain += 1
            else:
                correct_certain += 1
        else:
            if e > threshold:
                incorrect_uncertain += 1
            else:
                incorrect_certain += 1

    total = len(all_true)
    print(f"Total samples: {total}")
    print(f"Correct & Certain:   {correct_certain}")
    print(f"Correct & Uncertain: {correct_uncertain}")
    print(f"Incorrect & Certain: {incorrect_certain}")
    print(f"Incorrect & Uncertain: {incorrect_uncertain}")


def train_and_save_final_rf(df, top_features):
    X = df[top_features]
    y = df["label_num"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)

    # save final model and scaler
    joblib.dump(rf, "rf_uq_model_60.pkl")
    joblib.dump(scaler, "rf_scaler.save")


if __name__ == "__main__":
    loso_rf_with_dialogue(save_models=True)

    eda_df = pd.read_csv("preprocessing/data/eda_features_60.csv")
    ibi_df = pd.read_csv("preprocessing/data/ibi_features_60.csv")
    df = pd.merge(eda_df, ibi_df, on=["subject", "label", "window"])
    df["label_num"] = df["label"].map({"baseline": 0, "stress": 1})

    top_features = ['EDA_Tonic_SD', 'SCR_Peaks_Amplitude_Mean', 'SCR_Peaks_N', 'eda_kurtosis', 'eda_max', 'eda_mean', 'eda_min', 'eda_range', 'eda_skewness', 'eda_std', 'ibi_max', 'ibi_mean', 'ibi_median', 'ibi_min', 'ibi_range', 'ibi_std']

    train_and_save_final_rf(df, top_features)
