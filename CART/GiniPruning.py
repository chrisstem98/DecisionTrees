# ===============================================
# Bullet 3 – CART (Gini, pruning with ccp_alpha=0.1, with visualization)
# ===============================================

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Load the dataset ----------
base_dir = Path(__file__).parent
parent_dir = base_dir.parent.parent
file_path = parent_dir / "Raisin_Dataset.xlsx"

data = pd.read_excel(file_path)

# ---------- Separate features and target ----------
X = data.drop(columns=["Class"])
y = data["Class"]

# ---------- Create Decision Tree (CART) model ----------
# Criterion: 'gini'
# Pruning: ccp_alpha = 0.1
clf = DecisionTreeClassifier(
    criterion="gini",
    ccp_alpha=0.1,
    random_state=1
)

# ---------- 70/30 split evaluation ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=1
)
clf.fit(X_train, y_train)
acc_70_30 = clf.score(X_test, y_test) * 100.0

# ---------- Tree structure info ----------
num_leaves = clf.get_n_leaves()
tree_size = clf.tree_.node_count

# ---------- 10-fold Cross-Validation ----------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
acc_cv10 = cv_scores.mean() * 100.0

# ---------- Predictions ----------
y_pred = clf.predict(X_test)

# ---------- Confusion Matrix ----------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title("Confusion Matrix – CART (Gini, Pruned ccp_alpha=0.1)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# ---------- Feature Importance ----------
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": clf.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=feature_importance, x="Importance", y="Feature", palette="flare")
plt.title("Feature Importance – CART (Gini, Pruned ccp_alpha=0.1)")
plt.tight_layout()
plt.show()

# ---------- Tree Visualization ----------
plt.figure(figsize=(18, 9))
plot_tree(
    clf,
    filled=True,
    feature_names=X.columns,
    class_names=clf.classes_,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Visualization – CART (Gini, Pruned ccp_alpha=0.1)")
plt.show()

# ---------- Classification Report ----------
print("\n=== Classification Report (70/30 Split) ===")
print(classification_report(y_test, y_pred))

# ---------- Print Results ----------
print("\n=== Bullet 3 Results: CART (Gini, Pruning ccp_alpha=0.1) ===")
print(f"Accuracy (70/30 split): {acc_70_30:.4f}%")
print(f"Accuracy (10-fold CV): {acc_cv10:.4f}%")
print(f"Number of Leaves: {num_leaves}")
print(f"Tree Size (nodes): {tree_size}")
