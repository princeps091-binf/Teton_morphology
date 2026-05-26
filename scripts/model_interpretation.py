
%autoindent off
# %%
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna.visualization.matplotlib as ovis
import shap

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

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
# %%
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

# %%
loaded_model = lgb.Booster(model_file="./data/champion_cell_model.txt")

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


explainer = shap.TreeExplainer(loaded_model)
shap_values = explainer(X_raw_final)

# %%
sample_index = 73080
# %%
# 4. Generate the Force Plot
# matplotlib=True forces it to render a static image instead of HTML
shap.plots.force(
    explainer.expected_value, 
    shap_values.values[sample_index, :], 
    X_raw_final.iloc[sample_index, :], 
    matplotlib=True
)

plt.gcf().set_size_inches(16, 4)
plt.show()

# %%
shap.plots.waterfall(shap_values[sample_index])
