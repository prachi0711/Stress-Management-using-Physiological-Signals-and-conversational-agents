import copy
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

np.random.seed(42)
torch.manual_seed(42)

# Monte Carlo UQ
def mc_dropout_predict(model, xb, num_samples=50):
    model.train()  
    preds = []
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(xb)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds.append(probs)
    preds = np.stack(preds)
    mean_probs = preds.mean(axis=0)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
    return mean_probs, entropy

# 1D CNN
class Enhanced1DCNN(nn.Module):
    def __init__(self, input_features):
        super(Enhanced1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = nn.ReLU()(self.fc1(x))
        return self.fc2(x)

# training
def train_model_early_stopping(model, train_loader, val_loader, class_weights, max_epochs=50):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    best_val_loss = float('inf')
    patience = 0
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                outputs = model(xb)
                val_loss += criterion(outputs, yb).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= 5:
                break

    model.load_state_dict(best_model_wts)
    return model, best_epoch


def train_model_fixed_epochs(model, train_loader, class_weights, epochs):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
    return model

# LOSO evaluation
def loso(entropy_threshold=0.45):
    from dialogue_manager import DialogueManager 

    X = pd.read_csv(f"preprocessing/data/eda_features_60.csv")
    ibi = pd.read_csv(f"preprocessing/data/ibi_features_60.csv")
    label_df = pd.merge(X, ibi, on=["subject", "label", "window"])

    non_feature_cols = ['label', 'Label', 'subject', 'window']
    X = label_df.drop(columns=[col for col in non_feature_cols if col in label_df.columns], errors='ignore')
    y = label_df['label'].map({'baseline': 0, 'stress': 1}).values
    subjects = label_df['subject'].values

    # preprocess
    X = X.replace([np.inf, -np.inf], np.nan)
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    subjects = subjects[valid_mask]
    for col in X.columns:
        X[col].fillna(X[col].median(), inplace=True)
    X = X.loc[:, X.nunique() > 1]

    unique_subjects = np.unique(subjects)
    all_preds, all_true, all_entropy = [], [], []

    dm = DialogueManager(entropy_threshold=entropy_threshold)

    for test_subj in unique_subjects:
        test_idx = subjects == test_subj
        train_val_idx = ~test_idx
        X_train_val_df, y_train_val = X.iloc[train_val_idx], y[train_val_idx]
        X_test_df, y_test = X.iloc[test_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_val_scaled = scaler.fit_transform(X_train_val_df)
        X_test_scaled = scaler.transform(X_test_df)
        n_train_val = len(X_train_val_scaled)
        indices = np.arange(n_train_val)
        np.random.shuffle(indices)
        val_split = int(0.8 * n_train_val)
        train_idx, val_idx = indices[:val_split], indices[val_split:]
        X_train, y_train = X_train_val_scaled[train_idx], y_train_val[train_idx]
        X_val, y_val = X_train_val_scaled[val_idx], y_train_val[val_idx]

        # class weights
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        class_weights_np = 1.0 / class_counts
        class_weights_np = class_weights_np / class_weights_np.sum()
        class_weights_arr = np.zeros(np.max(unique_classes) + 1)
        for i, c in enumerate(unique_classes):
            class_weights_arr[c] = class_weights_np[i]
        class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32)

        # tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
        y_val = torch.tensor(y_val, dtype=torch.long)
        X_test = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y_test, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

        # training
        model = Enhanced1DCNN(input_features=X_train.shape[2])
        model, best_epoch = train_model_early_stopping(model, train_loader, val_loader, class_weights_tensor)

        scaler = StandardScaler()
        X_train_val_full = scaler.fit_transform(X_train_val_df)
        y_train_val_tensor = torch.tensor(y_train_val, dtype=torch.long)
        X_train_val_tensor = torch.tensor(X_train_val_full, dtype=torch.float32).unsqueeze(1)
        train_val_loader = DataLoader(TensorDataset(X_train_val_tensor, y_train_val_tensor), batch_size=32, shuffle=True)

        final_model = Enhanced1DCNN(input_features=X_train_val_tensor.shape[2])
        final_model = train_model_fixed_epochs(final_model, train_val_loader, class_weights_tensor, epochs=best_epoch)

        # evaluation with MC Dropout
        with torch.no_grad():
            for idx, (xb, yb) in enumerate(test_loader):
                mean_probs, entropy_vals = mc_dropout_predict(final_model, xb, num_samples=30)
                preds = np.argmax(mean_probs, axis=1)

                all_preds.extend(preds)
                all_true.extend(yb.cpu().numpy())
                all_entropy.extend(entropy_vals)

                for p, e in zip(preds, entropy_vals):
                    dm.get_response(user_id="user1", entropy_val=e, pred_label=p, sample_idx=idx)
        
    # Save model and scaler
        torch.save(model.state_dict(), "cnn_uq_model_60.pt")
        joblib.dump(scaler, "cnn_scaler.save")

    # UQ results
    report = classification_report(all_true, all_preds, target_names=["baseline", "stress"], output_dict=True)
    acc = accuracy_score(all_true, all_preds)

    correct_certain = 0
    correct_uncertain = 0
    incorrect_certain = 0
    incorrect_uncertain = 0

    for p, t, e in zip(all_preds, all_true, all_entropy):
        if p == t:
            if e > entropy_threshold:
                correct_uncertain += 1
            else:
                correct_certain += 1
        else:
            if e > entropy_threshold:
                incorrect_uncertain += 1
            else:
                incorrect_certain += 1

    print(f"Total: {len(all_true)}")
    print(f"Correct Certain: {correct_certain}")
    print(f"Correct Uncertain: {correct_uncertain}")
    print(f"Incorrect Certain: {incorrect_certain}")
    print(f"Incorrect Uncertain: {incorrect_uncertain}")


    return {
        "accuracy": acc,
        "stress_f1": report["stress"]["f1-score"],
        "baseline_f1": report["baseline"]["f1-score"]
    }

if __name__ == "__main__":

    loso(entropy_threshold=0.45)
    
