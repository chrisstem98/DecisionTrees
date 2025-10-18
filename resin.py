# ========================================
# Step 1 – Load Raisin Dataset from the Web
# ========================================
# All comments are in English.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Direct URL from the UCI Machine Learning Repository ---
# Get the current script's directory (folder2)
base_dir = Path(__file__).parent

# Go one level up to folder1
parent_dir = base_dir.parent

url = parent_dir / "Raisin_Dataset.xlsx"

# --- Read the Excel file directly from the URL ---
data = pd.read_excel(url)

# --- Show first few rows ---
print("=== First 5 Rows of the Raisin Dataset ===")
print(data.head(), "\n")

# --- Basic dataset information ---
print("=== Dataset Info ===")
print(data.info(), "\n")

# --- Statistical summary of numeric features ---
print("=== Statistical Summary ===")
print(data.describe(), "\n")

# --- Class distribution ---
print("=== Class Distribution ===")
print(data['Class'].value_counts(), "\n")

# --- Check for missing values ---
print("=== Missing Values per Column ===")
print(data.isnull().sum())

# =================================================
# Data Visualization – Histograms and Correlations
# =================================================

# --- 1. Histograms per feature, grouped by Class ---
num_cols = [col for col in data.columns if col != 'Class']

# Create histograms for each numeric feature
data[num_cols].hist(bins=20, figsize=(12, 8), edgecolor='black')
plt.suptitle("Feature Distributions – All Samples", fontsize=14)
plt.tight_layout()
plt.show()

# --- 2. Histogram per feature separated by Class ---
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=data, x=col, hue="Class", bins=20, kde=True, palette="viridis", alpha=0.6)
    plt.title(f"Distribution of {col} by Class")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# --- 3. Correlation Heatmap ---
plt.figure(figsize=(8, 6))
corr = data[num_cols].corr()
sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()