# ===============================================
# Bullet 4 – CART (Entropy, pruning with ccp_alpha=0.1)


import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# ---------- Load the dataset ----------
base_dir = Path(__file__).parent         
parent_dir = base_dir.parent   
parent_dir = parent_dir.parent 
file_path = parent_dir / "Raisin_Dataset.xlsx"

data = pd.read_excel(file_path)

# ---------- Separate features and target ----------
X = data.drop(columns=["Class"])
y = data["Class"]

# ---------- Create Decision Tree (CART) model ----------
# Criterion: 'entropy'
# Pruning: ccp_alpha = 0.1
clf = DecisionTreeClassifier(
    criterion="entropy",
    ccp_alpha=0.1,
    random_state=1
)

# ---------- 70/30 split evaluation ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=1
)
clf.fit(X_train, y_train)
acc_70_30 = clf.score(X_test, y_test) * 100.0

# Tree structure info
num_leaves = clf.get_n_leaves()
tree_size = clf.tree_.node_count

# ---------- 10-fold Cross-Validation ----------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
acc_cv10 = cv_scores.mean() * 100.0

# ---------- Print results ----------
print("=== Bullet 4 Results: CART (Entropy, Pruning ccp_alpha=0.1) ===")
print(f"Accuracy (70/30 split): {acc_70_30:.4f}%")
print(f"Accuracy (10-fold CV): {acc_cv10:.4f}%")
print(f"Number of Leaves: {num_leaves}")
print(f"Tree Size (nodes): {tree_size}")
