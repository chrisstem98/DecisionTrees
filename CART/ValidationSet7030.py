# ===============================================
# Ερώτημα 4 – Μέρος B: χωρίς validation set (10-fold CV)
# ===============================================

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# --- Load dataset ---
base_dir = Path(__file__).parent         
parent_dir = base_dir.parent   
parent_dir = parent_dir.parent 
file_path = parent_dir / "Raisin_Dataset.xlsx"

data = pd.read_excel(file_path)

# --- Split into training (70%) and test (30%) ---
X = data.drop(columns=["Class"])
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=1
)

# --- Compute candidate alphas from cost-complexity pruning path ---
clf = DecisionTreeClassifier(random_state=1, criterion="gini")
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# --- Evaluate each alpha using 10-fold cross-validation on training set ---
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
alpha_scores = []

for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(criterion="gini", ccp_alpha=alpha, random_state=1)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
    alpha_scores.append((alpha, np.mean(scores)))

alpha_scores = np.array(alpha_scores)
best_alpha_cv = alpha_scores[np.argmax(alpha_scores[:, 1]), 0]
best_cv_acc = alpha_scores[:, 1].max()

print("=== Part B: Cross-validation-based Pruning Selection ===")
print(f"Best alpha: {best_alpha_cv:.6f}")
print(f"Mean CV Accuracy: {best_cv_acc * 100:.2f}%")

# --- Train final model on full training set with best alpha ---
final_clf = DecisionTreeClassifier(criterion="gini", ccp_alpha=best_alpha_cv, random_state=1)
final_clf.fit(X_train, y_train)

# --- Evaluate on test set ---
test_acc_cv = final_clf.score(X_test, y_test) * 100
print(f"Test Accuracy (final model): {test_acc_cv:.2f}%")
