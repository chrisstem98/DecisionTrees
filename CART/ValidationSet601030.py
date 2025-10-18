# ===============================================
# Ερώτημα 4 – Μέρος A: με validation set (60%-10%-30%)
# ===============================================

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# --- Load dataset ---
base_dir = Path(__file__).parent         
parent_dir = base_dir.parent   
parent_dir = parent_dir.parent 
file_path = parent_dir / "Raisin_Dataset.xlsx"

data = pd.read_excel(file_path)

# --- Split into train (60%), validation (10%), test (30%) ---
X = data.drop(columns=["Class"])
y = data["Class"]

# 1st split: 70% train_val, 30% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=1
)

# 2nd split: from the 70%, take 60/10 (i.e., ~85.7% train, 14.3% val)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=(1/7), stratify=y_train_val, random_state=1
)

print("Training set size:", X_train.shape[0])
print("Validation set size:", X_val.shape[0])
print("Test set size:", X_test.shape[0])

# --- Step 1: Compute cost-complexity pruning path ---
clf = DecisionTreeClassifier(random_state=1, criterion="gini")
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# --- Step 2: Evaluate each alpha on validation set ---
results = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=1, criterion="gini", ccp_alpha=alpha)
    clf.fit(X_train, y_train)
    acc_val = clf.score(X_val, y_val)
    results.append((alpha, acc_val))

results = np.array(results)
best_alpha = results[np.argmax(results[:, 1]), 0]
best_acc = results[:, 1].max()

print("\n=== Part A: Validation-based Selection ===")
print(f"Best alpha: {best_alpha:.6f}")
print(f"Validation Accuracy: {best_acc*100:.2f}%")

# --- Step 3: Train final model on (train+val) with best alpha ---
X_train_combined = pd.concat([X_train, X_val])
y_train_combined = pd.concat([y_train, y_val])

final_clf = DecisionTreeClassifier(criterion="gini", ccp_alpha=best_alpha, random_state=1)
final_clf.fit(X_train_combined, y_train_combined)

# --- Step 4: Evaluate on test set ---
test_acc = final_clf.score(X_test, y_test) * 100
print(f"Test Accuracy (final model): {test_acc:.2f}%")
