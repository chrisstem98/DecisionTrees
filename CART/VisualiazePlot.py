# ===============================================
# Question 4 – Optimal alpha (CART cost-complexity pruning)
# ===============================================
# Includes both:
#   A) Validation set (60-10-30 split)
#   B) 10-fold Cross-Validation
#   and comparative plots
# ===============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# ---------- Load the dataset ----------
base_dir = Path(__file__).parent         
parent_dir = base_dir.parent   
parent_dir = parent_dir.parent 
file_path = parent_dir / "Raisin_Dataset.xlsx"

data = pd.read_excel(file_path)
X = data.drop(columns=["Class"])
y = data["Class"]

# ===============================================
# PART A: Validation set (60-10-30)
# ===============================================

# Split data: 60% train, 10% val, 30% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=1
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=(1 / 7), stratify=y_train_val, random_state=1
)

print("=== PART A: Validation-based Selection ===")
print(f"Training: {X_train.shape[0]} | Validation: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# Compute pruning path
clf = DecisionTreeClassifier(random_state=1, criterion="gini")
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas_A = path.ccp_alphas

# Evaluate each alpha on validation set
results_A = []
for alpha in ccp_alphas_A:
    clf = DecisionTreeClassifier(criterion="gini", ccp_alpha=alpha, random_state=1)
    clf.fit(X_train, y_train)
    acc_val = clf.score(X_val, y_val)
    results_A.append((alpha, acc_val))

results_A = np.array(results_A)
best_alpha_A = results_A[np.argmax(results_A[:, 1]), 0]
best_acc_A = results_A[:, 1].max()

print(f"Best alpha (A): {best_alpha_A:.6f}")
print(f"Validation Accuracy: {best_acc_A * 100:.2f}%")

# Train final model on (train+val)
X_train_combined = pd.concat([X_train, X_val])
y_train_combined = pd.concat([y_train, y_val])
final_clf_A = DecisionTreeClassifier(criterion="gini", ccp_alpha=best_alpha_A, random_state=1)
final_clf_A.fit(X_train_combined, y_train_combined)
test_acc_A = final_clf_A.score(X_test, y_test) * 100
print(f"Test Accuracy (final model): {test_acc_A:.2f}%\n")

# ===============================================
# PART B: Cross-Validation (70-30)
# ===============================================

print("=== PART B: Cross-validation-based Selection ===")

X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=1
)

clf = DecisionTreeClassifier(random_state=1, criterion="gini")
path = clf.cost_complexity_pruning_path(X_train_B, y_train_B)
ccp_alphas_B = path.ccp_alphas

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
alpha_scores_B = []
for alpha in ccp_alphas_B:
    clf = DecisionTreeClassifier(criterion="gini", ccp_alpha=alpha, random_state=1)
    scores = cross_val_score(clf, X_train_B, y_train_B, cv=cv, scoring="accuracy")
    alpha_scores_B.append((alpha, np.mean(scores)))

alpha_scores_B = np.array(alpha_scores_B)
best_alpha_B = alpha_scores_B[np.argmax(alpha_scores_B[:, 1]), 0]
best_acc_B = alpha_scores_B[:, 1].max()

print(f"Best alpha (B): {best_alpha_B:.6f}")
print(f"Mean CV Accuracy: {best_acc_B * 100:.2f}%")

# Train final model on full training set
final_clf_B = DecisionTreeClassifier(criterion="gini", ccp_alpha=best_alpha_B, random_state=1)
final_clf_B.fit(X_train_B, y_train_B)
test_acc_B = final_clf_B.score(X_test_B, y_test_B) * 100
print(f"Test Accuracy (final model): {test_acc_B:.2f}%\n")

# ===============================================
# PLOTS – Accuracy vs α for A and B
# ===============================================

plt.figure(figsize=(12, 5))

# --- Plot for Part A (Validation Set) ---
plt.subplot(1, 2, 1)
plt.plot(results_A[:, 0], results_A[:, 1] * 100, marker='o', linewidth=1.5)
plt.axvline(best_alpha_A, color='red', linestyle='--', label=f'Best α = {best_alpha_A:.5f}')
plt.title("Validation Accuracy vs ccp_alpha (Part A)")
plt.xlabel("ccp_alpha (α)")
plt.ylabel("Accuracy (%)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# --- Plot for Part B (Cross-Validation) ---
plt.subplot(1, 2, 2)
plt.plot(alpha_scores_B[:, 0], alpha_scores_B[:, 1] * 100, marker='o', linewidth=1.5)
plt.axvline(best_alpha_B, color='red', linestyle='--', label=f'Best α = {best_alpha_B:.5f}')
plt.title("10-fold CV Accuracy vs ccp_alpha (Part B)")
plt.xlabel("ccp_alpha (α)")
plt.ylabel("Mean CV Accuracy (%)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
