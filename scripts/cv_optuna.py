%autoindent off
# %%
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import os
import joblib
import optuna.visualization.matplotlib as ovis
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# %%


# =====================================================================
# 1. LIGHTWEIGHT PRE-COMPUTED FEATURE SELECTOR CLASS
# =====================================================================
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

# %%

# =====================================================================
# 3. SECURE LEAK-PROOF SPLIT
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

# %%

# =====================================================================
# 4. ONE-TIME PRE-COMPUTATION LAYER (CRUCIAL SPEEDUP)
# =====================================================================
print("\n[STEP 1/3] Running heavy calculations ONCE outside the optimization loop...")

# A. Calculate baseline Mutual Information ranking profile
print(" -> Profiling Feature Importance via Mutual Information...")
GLOBAL_MI_SCORES = mutual_info_classif(X_train, y_train, random_state=42)

# B. Build baseline correlation structure
print(" -> Generating absolute correlation distance matrix...")
corr_matrix = pd.DataFrame(X_train).corr().abs().to_numpy()
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

# =====================================================================
# 5. ULTRA-FAST OPTUNA PIPELINE OBJECTIVE
# =====================================================================
def objective(trial):
    # A. Parse Feature Selection Parameters
    linkage_method = trial.suggest_categorical('selector__linkage_method', ['ward', 'average'])
    criterion = 'distance'
    # Scale parameter space thresholds cleanly based on method physics
    if linkage_method == 'average':
        t = trial.suggest_float('selector__t', 0.1, 0.9)
    else: # ward
        t = trial.suggest_float('selector__t', 0.3, 2)
    # Fetch the relevant pre-computed linkage tree
    chosen_tree = LINKAGE_TREES[linkage_method]
    # B. Microsecond Slicing Sequence
    # Instantiates the selector using our static, pre-computed array blocks
    selector = PrecomputedClusterSelector(
        linkage_tree=chosen_tree, 
        mi_scores=GLOBAL_MI_SCORES, 
        criterion=criterion, 
        t=t
    )
    X_train_selected = selector.fit_transform(X_train)
    # Track selection counts natively
    trial.set_user_attr("n_features", X_train_selected.shape[1])
    # C. Configure LightGBM Hyperparameters
    lgbm_params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_jobs': 1, # Prevent core thrashing on workers
        'num_threads':1,
        'num_leaves': trial.suggest_int('lgb__num_leaves', 15, 63),
        'tree_method': 'hist',
        'max_depth': trial.suggest_int('lgb__max_depth', 3, 10),
        'learning_rate': trial.suggest_float('lgb__learning_rate', 0.025, 0.1, log=True),
        'feature_fraction': trial.suggest_float('lgb__feature_fraction', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('lgb__min_child_samples', 20, 100),
        'n_estimators': 5000}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_auc_scores = []
    for train_idx, val_idx in skf.split(X_train_selected, y_train):
        # Slice your data natively
        X_tr, X_val = X_train_selected.iloc[train_idx], X_train_selected.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        # Initialize the native scikit-learn API model
        model = lgb.LGBMClassifier(**lgbm_params)
        # Fit sequentially with evaluation metrics and early stopping
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        # Calculate this fold's peak performance
        val_preds = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_auc_scores.append(fold_auc)
        del model
    mean_score = np.mean(fold_auc_scores)
    del X_train_selected, fold_auc_scores
    gc.collect()
    return mean_score

# %%

# =====================================================================
# 6. RUNNING THE SPEED-OPTIMIZED STUDY
# =====================================================================
os.environ["OMP_NUM_THREADS"] = "1"
print("\n[STEP 2/3] Initializing Optuna Optimization Loop...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20,show_progress_bar=True,n_jobs=5)

print("\n=== DESIGN MATRIX RUNTIME SUMMARY ===")
print(f"Peak Internal Validation AUC: {study.best_value:.4f}")
print(f"Optimal Parameters: {study.best_params}")
print(f"Winning Representative Feature Count: {study.best_trial.user_attrs['n_features']}")

# %%

study.trials_dataframe().to_csv('./data/distance_only_trials.txt',index=False,sep='\t')
# Save the Optuna Study object
# We use joblib because pickle/joblib preserves the full internal study history
study_filename = "./data/optuna_morphology_study.pkl"
joblib.dump(study, study_filename)
# %%

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
ovis.plot_contour(study, params=["selector__", "lgb__num_leaves"])
plt.show()
# %%
# =====================================================================
# 7. PRODUCTION RE-FIT & HOLDOUT EVALUATION
# =====================================================================

# 1. Reconstruct the Winning Selector using the Best Parameters
# Hard-coding 'distance' since we proved it prevents the 1-feature trap
best_linkage = study.best_params['selector__linkage_method']
best_t = study.best_params['selector__t']

final_selector = PrecomputedClusterSelector(
    linkage_tree=LINKAGE_TREES[best_linkage],
    mi_scores=GLOBAL_MI_SCORES,
    criterion='distance',
    t=best_t
)

# 2. Transform BOTH datasets using the winning architecture
X_train_final = final_selector.fit_transform(X_train)
X_test_final = final_selector.transform(X_test)

n_selected_features = X_train_final.shape[1]

# %%
# 3. Slice an explicit 10% validation pool out of the training set for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train, test_size=0.10, random_state=42, stratify=y_train
)

# 4. Format data streams into raw, single-threaded LightGBM Datasets
dtrain_final = lgb.Dataset(X_tr.to_numpy(), label=y_tr)
dval_final = lgb.Dataset(X_val.to_numpy(), label=y_val, reference=dtrain_final)

# %%
# 5. Extract and Map the Optimized LightGBM Tree Parameters
final_lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'num_threads': 1,  # Keep single-threaded stability lock
    'learning_rate': study.best_params['lgb__learning_rate'],
    'num_leaves': study.best_params['lgb__num_leaves'],
    'max_depth': study.best_params['lgb__max_depth'],
    'feature_fraction': study.best_params['lgb__feature_fraction'],
    'min_child_samples': study.best_params['lgb__min_child_samples'],
 }

# %%
print("\nTraining final booster...")
champion_booster = lgb.train(
    final_lgbm_params,
    dtrain_final,
    num_boost_round=5000,
    valid_sets=[dval_final],
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
)
print(f" -> Active convergence reached at tree #{champion_booster.best_iteration}")

# 6. RUN INFERENCE ON THE UNTOUCHED HOLDOUT TEST DATASET
# (This acts as the ultimate truth test for data leakage)
test_probabilities = champion_booster.predict(
    X_test_final.to_numpy(), 
    num_iteration=champion_booster.best_iteration
)
# %%
# Convert continuous probabilities back to binary class names
test_predictions_binary = (test_probabilities > 0.5).astype(int)
test_predictions_strings = label_encoder.inverse_transform(test_predictions_binary)
y_test_strings = label_encoder.inverse_transform(y_test)

# 7. METRIC READOUT PRESENTATION
final_holdout_auc = roc_auc_score(y_test, test_probabilities)

print(classification_report(y_test_strings, test_predictions_strings))

# %%
model_filename = "./data/champion_cell_model.txt"
champion_booster.save_model(model_filename)
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
