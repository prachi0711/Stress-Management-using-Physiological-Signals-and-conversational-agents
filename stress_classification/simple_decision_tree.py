import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "preprocessing/data/"  
WINDOW_SIZES = [30, 45, 60, 75, 90, 120]
RANDOM_STATE = 42

def load_and_combine_features(window_size):
    try:
        eda_df = pd.read_csv(os.path.join(DATA_DIR, f"eda_features_{window_size}.csv"))
        ibi_df = pd.read_csv(os.path.join(DATA_DIR, f"ibi_features_{window_size}.csv"))
        
        combined = pd.merge(
            eda_df, 
            ibi_df,
            on=['subject', 'window', 'label'],
            suffixes=('_eda', '_ibi')
        )
        
        combined = combined.dropna()
        
        # encoded labels ('baseline'/'stress' to 0/1)
        combined['label_encoded'] = combined["label"].map({"baseline": 0, "stress": 1})
        combined['label'] = combined['label'].astype('category') 
        
        return combined
    
    except FileNotFoundError:
        print(f"Warning: Missing files for window size {window_size}s")
        return None

def evaluate_window_size(df, window_size):
    if df is None or len(df) == 0:
        return {
            'window_size': window_size,
            'status': 'FAILED (no valid data)'
        }
    
    X = df.drop(['subject', 'window', 'label', 'label_encoded'], axis=1)
    y = df['label_encoded']
    groups = df['subject']
    
    # LOSO approach
    logo = LeaveOneGroupOut()
    reports = []
    failed_folds = 0
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        if len(np.unique(y_test)) < 2:
            failed_folds += 1
            continue
            
        model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=5,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        report = classification_report(
            y_test, 
            y_pred, 
            output_dict=True,
            zero_division=0,
            target_names=['baseline', 'stress'] 
        )
        reports.append(report)
    
    if reports:
        return {
            'window_size': window_size,
            'accuracy': np.mean([r['accuracy'] for r in reports]),
            'stress_f1': np.mean([r['stress']['f1-score'] for r in reports]),
            'baseline_f1': np.mean([r['baseline']['f1-score'] for r in reports]),
        }
    else:
        return {
            'window_size': window_size,
            'status': f'FAILED ({failed_folds} folds had single-class)'
        }

def main():
    results = []
    
    for size in WINDOW_SIZES:
        
        df = load_and_combine_features(size)
        scores = evaluate_window_size(df, size)
        results.append(scores)
        
    results_df = pd.DataFrame(results)
    
    print("\n Results")
    print(results_df[[
        'window_size', 'accuracy', 
        'stress_f1', 'baseline_f1'
    ]].to_string(index=False))

if __name__ == "__main__":
    main()