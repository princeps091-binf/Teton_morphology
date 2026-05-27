
%autoindent off
# %%
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna.visualization.matplotlib as ovis
import shap
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# %%

class PrecomputedClusterSelector(BaseEstimator, TransformerMixin):
    """
    Slices a pre-computed hierarchical linkage tree and uses pre-computed 
    Mutual Information scores to select the best representative feature 
    per cluster block instantaneously.
    """
    def __init__(self, linkage_tree, mi_scores, criterion='distance', t=0.5):
        self.linkage_tree = linkage_tree
        self.mi_scores = mi_scores
        self.criterion = criterion
        self.t = t
        self.selected_features_ = None
    def fit(self, X, y=None):
        num_features = X.shape[1]
        # 1. Instantly slice the pre-computed linkage tree using threshold 't'
        cluster_ids = hierarchy.fcluster(
            self.linkage_tree, t=self.t, criterion=self.criterion)
        # 2. Group features by cluster blocks
        feature_clusters = pd.DataFrame({
            'feature_idx': range(num_features),
            'cluster_id': cluster_ids,
            'mi_score': self.mi_scores
        })
        # 3. Extract the single feature with the highest MI score per cluster
        best_features = (
            feature_clusters
            .sort_values(by='mi_score', ascending=False)
            .groupby('cluster_id')
            .first())
        self.selected_features_ = best_features['feature_idx'].tolist()
        return self
    def transform(self, X):
        return X.iloc[:, self.selected_features_]



# %%


data_tbl = pd.read_csv('~/Documents/Teton_project/data/morphology_metadata_TETON.csv')
treatment_labels = ['Ctrl','40nMImatinib']

modeling_data_tbl = data_tbl.query('treatment in @treatment_labels')

columns_select = []
#columns_select.extend(['Cell'])
columns_select.extend(list(data_tbl.columns[11:592]))

X_raw = modeling_data_tbl.loc[:,columns_select] 
y_raw = modeling_data_tbl.treatment.to_numpy()
# 1. Initialize and fit the LabelEncoder on your raw string targets
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw) # Converts strings to 0 and 1
class_mapping = label_encoder.classes_


GLOBAL_MI_SCORES = mutual_info_classif(X_raw, y_encoded, random_state=42)

# B. Build baseline correlation structure
print(" -> Generating absolute correlation distance matrix...")
corr_matrix = pd.DataFrame(X_raw).corr().abs().to_numpy()
corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
distance_matrix = np.clip(1 - corr_matrix, 0, 1)
condensed_distances = squareform(distance_matrix)

# C. Generate pre-computed linkage trees (Ward & Average methods)
print(" -> Fitting Hierarchical Cluster Trees...")

LINKAGE_TREES = {
    'ward': hierarchy.ward(condensed_distances),
    'average': hierarchy.average(condensed_distances)
}
print("Pre-computation phase complete. Pipeline is primed.")


# %%

study = joblib.load("./data/optuna_morphology_study.pkl")

print("--- Loaded Study Summary ---")
print(f"Best Trial Number   : {study.best_trial.number}")
print(f"Best Validation AUC : {study.best_value:.4f}")
print(f"Winning Parameters  : {study.best_params}")

ovis.plot_param_importances(study)
plt.show()

import optuna.importance as importance

# Calculate the mathematical impact of each parameter
param_importances = importance.get_param_importances(study)

print("=== Relative Hyperparameter Drivers ===")
for param, val in param_importances.items():
    print(f"{param:<35}: {val*100:>5.1f}% impact")

# %%
import optuna.visualization.matplotlib as ovis
ovis.plot_param_importances(study)
plt.tight_layout()
plt.show()

# %%

ovis.plot_optimization_history(study)
plt.show()

# %%
ovis.plot_contour(study, params=["lgb__feature_fraction", "lgb__max_depth"])
plt.show()

# %%
loaded_model = lgb.Booster(model_file="./data/champion_cell_model.txt")

# %%

# %%

best_linkage = study.best_params['selector__linkage_method']
best_t = study.best_params['selector__t']

final_selector = PrecomputedClusterSelector(
    linkage_tree=LINKAGE_TREES[best_linkage],
    mi_scores=GLOBAL_MI_SCORES,
    criterion='distance',
    t=best_t
)

# 2. Transform BOTH datasets using the winning architecture
X_raw_final = final_selector.fit_transform(X_raw)

# %%

test_probabilities = loaded_model.predict(
    X_raw_final.to_numpy(), 
    num_iteration=loaded_model.best_iteration
)

# %%
test_predictions_binary = (test_probabilities > 0.992).astype(int)
con_mat = confusion_matrix(y_encoded, test_predictions_binary)

class_names = label_encoder.classes_
# 4. Initialize the Matplotlib figure canvas
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

# 5. Build and format the Confusion Matrix Display object
# We use the clean 'Blues' colormap to mirror professional screening layouts
disp = ConfusionMatrixDisplay(
        confusion_matrix=con_mat, 
    display_labels=class_names
)
# Render the matrix onto our specified axis, removing the default scikit-learn colorbar
disp.plot(
    cmap=plt.cm.Blues, 
    ax=ax, 
    values_format='d',  # 'd' forces integers instead of scientific notation
    colorbar=False      
)

# 6. Customize Fine Typography & Labeling Aesthetics
ax.set_title('Confusion Matrix Readout\nIndependent Holdout Test Split', fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Predicted Label Designation', fontsize=11, fontweight='bold', labelpad=10)
ax.set_ylabel('True Biological Label (Ground Truth)', fontsize=11, fontweight='bold', labelpad=10)

# Clean up tick label presentation
ax.set_xticklabels(class_names, fontsize=10)
ax.set_yticklabels(class_names, fontsize=10, rotation=90, va="center")

# 7. Save the high-resolution vector graphic to your local workspace
#output_matrix_path = "model_performance_confusion_matrix.png"
#plt.savefig(output_matrix_path, dpi=300, bbox_inches='tight')
#print(f" -> Confusion Matrix saved to disk as: {output_matrix_path}")
plt.show()
# %%
fpr, tpr, thresholds = roc_curve(y_encoded, test_probabilities)


# 3. Calculate the exact area under the curve (AUC) metric for the plot legend
roc_auc = auc(fpr, tpr)

# 4. Initialize the Matplotlib figure canvas
plt.figure(figsize=(15, 12), dpi=300)

# 5. Plot the Champion Model's ROC Curve
plt.plot(
    fpr, 
    tpr, 
    color='darkorange', 
    lw=2.5, 
    label=f'LightGBM Model (AUC = {roc_auc:.4f})'
)

# 6. Plot the No-Skill Baseline (The diagonal random guess line)
plt.plot(
    [0, 1], 
    [0, 1], 
    color='navy', 
    lw=1.5, 
    linestyle='--', 
    label='Random Classification Baseline (AUC = 0.5000)'
)

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=5, fontweight='bold', labelpad=5)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=5, fontweight='bold', labelpad=5)
plt.title('Receiver Operating Characteristic (ROC) Curve\nHoldout Test Dataset Evaluation', fontsize=8, fontweight='bold', pad=15)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc="lower right", fontsize=6, frameon=True, shadow=False)

# 8. Save the high-resolution vector graphic to your local repository
#output_image_path = "model_performance_roc_curve.png"
#plt.savefig(output_image_path, dpi=300, bbox_inches='tight')

# 9. Display the figure on-screen
plt.show()


# %%

explainer = shap.TreeExplainer(loaded_model)
shap_values = explainer(X_raw_final)

# %%
sample_index = 308

shap.plots.waterfall(shap_values[sample_index])

# %%
# If shap value > 0 contributes to be in label 1
# if shap value < 0 contributes to be in label 0

tmp_ax = pd.DataFrame({'shap':shap_values.mean(axis=1).values,'y':y_encoded,'prob':test_probabilities}).groupby('y').shap.plot.kde(legend=True)

plt.show()

# %%

# The certainty landscape for individual cells
tmp_ax = pd.DataFrame({'shap':shap_values.mean(axis=1).values,'y':y_encoded,'prob':test_probabilities}).assign(cert = lambda df: 2 * np.abs(0.5 - df.prob)).plot.scatter(x='shap',y='cert',logy=True,alpha=0.2)
plt.show()

# %%

tmp_idx =71500
from kneed import KneeLocator
tmp_cell = shap_values[tmp_idx,:]
tmp_cell_label = (2*y_encoded[tmp_idx] - 1)
tmp_correct_direction_features = tmp_cell[(tmp_cell.values * tmp_cell_label) > 0]

tmp_ax = pd.DataFrame({'shap':np.abs(tmp_correct_direction_features.values)}).assign(shap_rank = lambda df: df.shap.rank(pct=True,ascending=False)).sort_values('shap_rank').plot(x='shap_rank',y='shap')

plt.show()

df_fit, loc_fit, scale_fit = stats.t.fit(tmp_cell.values,floc=0)

p_local = 2 * stats.t.sf(np.abs(tmp_correct_direction_features.values), df_fit, loc=0, scale=scale_fit)

np.min(p_local)
# %%

cluster_ids = hierarchy.fcluster(LINKAGE_TREES[best_linkage], t=best_t, criterion='distance')
reordered_indices = hierarchy.leaves_list(LINKAGE_TREES[best_linkage])
max(cluster_ids)
# Reorder the correlation matrix symmetrically
reordered_corr = corr_matrix[reordered_indices, :][:, reordered_indices]
reordered_clusters = cluster_ids[reordered_indices]

# %%
def make_shuffled_cmap(num_categories=150, base_cmap='turbo', seed=42):
    # 1. Sample evenly spaced colors from a high-spectrum continuous map
    base = plt.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, num_categories))
    # 2. Shuffle the colors to maximize contrast between adjacent numbers
    rng = np.random.default_rng(seed)
    rng.shuffle(color_list)
    # 3. Return as a discrete ListedColormap
    return ListedColormap(color_list)

# %%
# Generate a 120-category custom colormap
my_huge_cmap = make_shuffled_cmap(num_categories=max(cluster_ids +1 ))

# %%
fig, (ax_cluster, ax_heatmap) = plt.subplots(
    1, 2, 
    figsize=(9, 7), 
    sharey=True, 
    gridspec_kw={'width_ratios': [0.04, 1], 'wspace': 0.02},
    layout='tight'
)
cluster_vector = reordered_clusters.reshape(-1, 1)
# 'tab10' or 'Set3' discrete colormaps work best for clean category separation
ax_cluster.imshow(cluster_vector, cmap=my_huge_cmap, aspect='auto')
ax_cluster.set_xticks([])
ax_cluster.set_ylabel("Features")
ax_cluster.spines[:].set_visible(False)
# matshow/imshow are incredibly fast even for thousands of elements
cax = ax_heatmap.imshow(reordered_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='equal')
# HIDE LABELS: Pass an empty list to the tick label setters
ax_heatmap.set_xticklabels([])  # Removes column labels
ax_heatmap.set_yticklabels([])  # Removes row labels (optional)
# Add a colorbar matching the scale exactly
fig.colorbar(cax, ax=ax_heatmap, orientation='vertical', shrink=0.8)
plt.show()
