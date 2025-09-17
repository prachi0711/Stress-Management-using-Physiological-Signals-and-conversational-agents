import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

window_sizes = [30, 45, 60, 75, 90, 120]
k_values = [5, 10, 15, 16] # k best features, 16 is total number of features

final_results = []

for window_size in window_sizes:
    print(f"\n Processing Window Size: {window_size}")

    eda_path = f"preprocessing/data/eda_features_{window_size}.csv"
    ibi_path = f"preprocessing/data/ibi_features_{window_size}.csv"

    if not os.path.exists(eda_path) or not os.path.exists(ibi_path):
        print(f"Missing files for window size {window_size}, skipping...")
        continue

    eda_df = pd.read_csv(eda_path)
    ibi_df = pd.read_csv(ibi_path)
    df = pd.merge(eda_df, ibi_df, on=["subject", "label", "window"])

    # encode labels ('baseline'/'stress' to 0/1)
    df["label_num"] = df["label"].map({"baseline": 0, "stress": 1})

    subjects = df["subject"].unique()
    
    for k in k_values:
        print(f"\n-- Evaluating with top {k} global features --")

        X = df.drop(columns=["subject", "label", "label_num", "window"])
        y = df["label_num"]

        X = X.loc[:, X.nunique() > 1]

        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        selector.fit(X, y)

        selected_features = X.columns[selector.get_support()]
        print(f"Selected features: {sorted(selected_features)}")

        # LOSO approach
        all_true = []
        all_pred = []

        for test_subj in subjects:
            train_df = df[df["subject"] != test_subj]
            test_df = df[df["subject"] == test_subj]

            X_train = train_df[selected_features]
            y_train = train_df["label_num"]
            X_test = test_df[selected_features]
            y_test = test_df["label_num"]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

            all_true.extend(y_test)
            all_pred.extend(y_pred)

        acc = accuracy_score(all_true, all_pred)
        report = classification_report(all_true, all_pred, output_dict=True, zero_division=0)
        
        result_entry = {
            'window_size': window_size,
            'k_features': k,
            'accuracy': acc,
            'stress_f1': report['1']['f1-score'] if '1' in report else 0,
            'baseline_f1': report['0']['f1-score'] if '0' in report else 0,
        }
        final_results.append(result_entry)
        
        print(f"Accuracy with top {k} features: {acc:.4f}")
        print(f"Stress F1: {result_entry['stress_f1']:.4f}")
        print(f"Baseline F1: {result_entry['baseline_f1']:.4f}")

results_df = pd.DataFrame(final_results)

plt.figure(figsize=(8, 6))
sns.lineplot(
    data=results_df,
    x="k_features",
    y="accuracy",
    hue="window_size",
    style="window_size",
    markers=True,
    dashes=False,
    palette="tab10",   
    linewidth=2.2
)

plt.title("Accuracy vs Number of Features")
plt.xlabel("Number of Selected Features (k)")
plt.ylabel("Accuracy")
plt.legend(title="Window Size (s)")
plt.grid(True)
plt.tight_layout()
plt.show()

print("Result BY window size (Best k for each window)")
summary_by_window = results_df.loc[results_df.groupby('window_size')['accuracy'].idxmax()]
print(summary_by_window[['window_size', 'k_features', 'accuracy', 'stress_f1', 'baseline_f1']].to_string(index=False))