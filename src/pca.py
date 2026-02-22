# ======================================================================
# PCA ON ACOUSTIC FEATURES
#
# INPUT:  Excel file with columns:
#   main_folder, file_name, relative_path,
#   duration_s, snr_db, f0_mean_hz, ..., mfcc3_std, voiced_ratio
#
# OUTPUTS:
#   1. Scree plot (png)
#   2. PCA explained variance table (xlsx)
#   3. PCA loadings table (features x PCs) (xlsx)
#   4. PC1 vs PC2 scatter plot (png)
#   5. PC1 vs PC2 biplot with feature arrows (png)
#
# ======================================================================

# 0. Install and import libraries
#pip install -q pandas numpy matplotlib scikit-learn openpyxl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["font.size"] = 11



# Read Excel into DataFrame
df = pd.read_csv("acoustic_features_24.csv")  # Change to pd.read_excel(file_name) if it's an Excel file


# ======================================================================
# 2. Split metadata vs numeric feature columns
# ======================================================================

# All non-numeric columns (object, string, etc.) are treated as metadata
meta_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print("\nMetadata (non-numeric) columns (NOT used in PCA):")
print(meta_cols)

# All numeric columns are candidates for PCA features
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumeric feature columns (used in PCA):")
print(feature_cols)

# Extract metadata and feature matrices
meta_df = df[meta_cols] if meta_cols else pd.DataFrame(index=df.index)
X = df[feature_cols].copy()

# ======================================================================
# 3. Handle missing values & standardize features
# ======================================================================

# Impute missing values (if any) with median of each feature
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Standardize features to mean=0, std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print("\nShape of feature matrix (n_samples, n_features):", X_scaled.shape)

# ======================================================================
# 4. Run PCA (all components)
# ======================================================================

pca = PCA(n_components=None)
X_pca = pca.fit_transform(X_scaled)

n_components = pca.n_components_
print(f"\nPCA done. Number of components = {n_components}")

# Explained variance info
explained_var_ratio = pca.explained_variance_ratio_
explained_var = pca.explained_variance_
cum_explained = np.cumsum(explained_var_ratio)

pc_labels = [f"PC{i+1}" for i in range(n_components)]
ev_df = pd.DataFrame({
    "PC": pc_labels,
    "Eigenvalue": explained_var,
    "Proportion_of_Variance": explained_var_ratio,
    "Cumulative_Variance": cum_explained
})

print("\nExplained variance table:")

# Save explained variance table to Excel
ev_file = "pca_explained_variance.xlsx"
ev_df.to_excel(ev_file, index=False)
print(f"Saved explained variance table to {ev_file}")

# ======================================================================
# 5. Scree plot (individual + cumulative variance)
# ======================================================================

plt.figure()
x = np.arange(1, n_components + 1)

plt.plot(x, explained_var_ratio, marker="o", label="Individual variance")
plt.plot(x, cum_explained, marker="s", linestyle="--", label="Cumulative variance")

plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.title("Scree Plot (PCA on Acoustic Features)")
plt.xticks(x)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

scree_file = "pca_scree_plot.png"
plt.tight_layout()
plt.savefig(scree_file, dpi=300, bbox_inches="tight")
print(f"\nSaved scree plot to {scree_file}")
plt.show()

# ======================================================================
# 6. PCA loadings (feature contributions to each PC)
# ======================================================================

loadings = pd.DataFrame(
    pca.components_.T,
    index=feature_cols,
    columns=pc_labels
)

print("\nPCA loadings (first few features):")

loadings_file = "pca_loadings.xlsx"
loadings.to_excel(loadings_file)
print(f"Saved loadings table to {loadings_file}")

# ======================================================================
# 7. PC1 vs PC2 scatter plot
# ======================================================================

# Create a DataFrame with the first two PCs
pca_scores_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
combined_df = pd.concat([meta_df, pca_scores_df], axis=1)

print("\nPCA scores (first few rows):")

has_class = "ClassLabel" in combined_df.columns

plt.figure()
if has_class:
    classes = combined_df["ClassLabel"].unique()
    for cls in classes:
        subset = combined_df[combined_df["ClassLabel"] == cls]
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            label=str(cls),
            alpha=0.7
        )
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
else:
    plt.scatter(combined_df["PC1"], combined_df["PC2"], alpha=0.7)

plt.axhline(0, color="grey", linewidth=0.8)
plt.axvline(0, color="grey", linewidth=0.8)
plt.xlabel(f"PC1 ({explained_var_ratio[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({explained_var_ratio[1]*100:.1f}% var)")
plt.title("PC1 vs PC2 Scores (Acoustic Features)")
plt.grid(True, linestyle="--", alpha=0.4)

scores_file = "pca_scores_PC1_PC2.png"
plt.tight_layout()
plt.savefig(scores_file, dpi=300, bbox_inches="tight")
print(f"\nSaved PC1 vs PC2 scores plot to {scores_file}")
plt.show()

# ======================================================================
# 8. PC1 vs PC2 biplot (scores + feature loading vectors)
# ======================================================================

plt.figure()

# Plot points (scores)
if has_class:
    classes = combined_df["ClassLabel"].unique()
    for cls in classes:
        subset = combined_df[combined_df["ClassLabel"] == cls]
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            alpha=0.5,
            label=str(cls)
        )
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
else:
    plt.scatter(combined_df["PC1"], combined_df["PC2"], alpha=0.5)

# Loadings for PC1 and PC2
pc1_loadings = loadings["PC1"].values
pc2_loadings = loadings["PC2"].values
arrow_scale = 3.0

for i, feature in enumerate(feature_cols):
    x_arrow = pc1_loadings[i] * arrow_scale
    y_arrow = pc2_loadings[i] * arrow_scale
    plt.arrow(0, 0, x_arrow, y_arrow,
              color="red", alpha=0.7,
              head_width=0.05, length_includes_head=True)
    plt.text(x_arrow * 1.05, y_arrow * 1.05, feature,
             color="red", fontsize=8)

plt.axhline(0, color="grey", linewidth=0.8)
plt.axvline(0, color="grey", linewidth=0.8)
plt.xlabel(f"PC1 ({explained_var_ratio[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({explained_var_ratio[1]*100:.1f}% var)")
plt.title("PCA Biplot: PC1 vs PC2 (Scores + Loadings)")
plt.grid(True, linestyle="--", alpha=0.4)

biplot_file = "pca_biplot_PC1_PC2.png"
plt.tight_layout()
plt.savefig(biplot_file, dpi=300, bbox_inches="tight")
print(f"\nSaved biplot to {biplot_file}")
plt.show()

print("\nPCA analysis complete.")