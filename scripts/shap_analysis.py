%autoindent off

# %%
import pandas as pd
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from matplotlib.colors import ListedColormap
from pathos.pools import ProcessPool
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests
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
corr_matrix = pd.DataFrame(X_raw).corr().abs().to_numpy()
corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
distance_matrix = np.clip(1 - corr_matrix, 0, 1)
condensed_distances = squareform(distance_matrix)

# C. Generate pre-computed linkage trees (Ward & Average methods)

LINKAGE_TREES = {
    'ward': hierarchy.ward(condensed_distances),
    'average': hierarchy.average(condensed_distances)
}

# %%
loaded_model = lgb.Booster(model_file="./data/champion_cell_model.txt")

study = joblib.load("./data/optuna_morphology_study.pkl")

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

test_probabilities = loaded_model.predict(
    X_raw_final.to_numpy(), 
    num_iteration=loaded_model.best_iteration
)

# %%

explainer = shap.TreeExplainer(loaded_model)
shap_values = explainer(X_raw_final)

# %%
# If shap value > 0 contributes to be in label 1
# if shap value < 0 contributes to be in label 0

tmp_ax = pd.DataFrame({'shap':shap_values.mean(axis=1).values,'y':y_encoded,'prob':test_probabilities}).groupby('y').shap.plot.kde(legend=True)

plt.show()

# %%

from kneed import KneeLocator, find_shape

# %%
GLOBAL_SHAP_MATRIX = shap_values.values
GLOBAL_FEATURE_NAMES = list(shap_values.feature_names)

# Lock down the parent view as read-only for strict Linux Copy-on-Write safety
GLOBAL_SHAP_MATRIX.flags.writeable = False
# =====================================================================
# 1. DEFINE THE FORK-SAFE WORKER
# =====================================================================
def process_local_cell_worker(tmp_idx,y_encoded_val,y_raw_val):
    """
    Worker function for Linux (fork). It reads directly from the 
    GLOBAL_SHAP_MATRIX variable without copying any data.
    """
    # -----------------------------------------------------------------
    # TRUE ZERO-COPY: Accessing the global variable directly from parent RAM
    # -----------------------------------------------------------------
    tmp_cell_values = GLOBAL_SHAP_MATRIX[tmp_idx, :]
    # Fit the local zero-centered t-distribution to adapt to certainty stretching
    df_fit, loc_fit, scale_fit = stats.t.fit(tmp_cell_values, floc=0)
    # Determine directional alignment with the predicted label
    tmp_cell_label = (2 * y_encoded_val - 1)
    aligned_mask = (tmp_cell_values * tmp_cell_label) > 0
    if not np.any(aligned_mask) or len(tmp_cell_values) < 3:
        return None
    # Isolate congruent features
    correct_values = tmp_cell_values[aligned_mask]
    correct_names = [GLOBAL_FEATURE_NAMES[i] for i, flag in enumerate(aligned_mask) if flag]
    # Compute local two-tailed tail probabilities using THIS cell's unique scale
    p_local = 2 * stats.t.sf(np.abs(correct_values), df_fit, loc=0, scale=scale_fit)
    # Build tracking frame
    tmp_cell_tbl = pd.DataFrame({
        'shap': np.abs(correct_values),
        'feature_name': correct_names,
        'pvalue': p_local
    })
    # Calculate percentage-based rank
    tmp_cell_tbl['shap_rank'] = tmp_cell_tbl['shap'].rank(pct=True, ascending=False)
    tmp_cell_tbl = tmp_cell_tbl.sort_values('shap_rank')
    x_data = tmp_cell_tbl['shap_rank'].to_numpy()
    y_data = tmp_cell_tbl['shap'].to_numpy()
    try:
        direction, curve = find_shape(x_data, y_data)
        kneedle = KneeLocator(x=x_data, y=y_data, curve=curve, direction=direction, S=1.0)
        if kneedle.knee is not None:
            cutoff = kneedle.knee
            tmp_cell_top_feature_tbl = tmp_cell_tbl.query('shap_rank < @cutoff').copy()
            # Label with metadata
            tmp_cell_top_feature_tbl['cell_idx'] = tmp_idx
            tmp_cell_top_feature_tbl['label'] = y_raw_val
            return tmp_cell_top_feature_tbl
    except Exception:
        pass
    return None

# %%
n_cells = GLOBAL_SHAP_MATRIX.shape[0]

# Prepare individual argument vectors for pathos (it handles multi-argument maps beautifully)
indices = list(range(n_cells))
encoded_labels = list(y_encoded)
raw_labels = list(y_raw)

print(f"Spawning Pathos ProcessPool on Linux...")
# Pathos handles pool creation and context allocation automatically
pool = ProcessPool()
# Map the inputs across your workers
results = pool.map(process_local_cell_worker, indices, encoded_labels, raw_labels)
print("Compiling final population ledger...")
shap_res_df = pd.concat([df for df in results if df is not None])
# %%

tmp_ax = (shap_res_df
          .merge(pd.DataFrame({'cell_idx':list(range(test_probabilities.shape[0])),'proba':test_probabilities}))
#          .query('~(label == "Ctrl")')
          .query('pvalue < 0.5')
          .feature_name.value_counts()
          .reset_index().assign(crank = lambda df: df.loc[:,'count'].rank(pct=True,ascending=False))
          .plot.scatter(x='crank',y='count')
          )
plt.show()

# %%
# 5. Compute Universal Binomial Grid Parameters
total_cells = X_raw_final.shape[0]
total_active_pool_features = X_raw_final.shape[1]
total_possible_experiment_slots = total_cells * total_active_pool_features
grand_total_observed_hits = shap_res_df.shape[0] 

# Baseline probability of a cell observation slot firing by chance
global_p_baseline = grand_total_observed_hits / total_possible_experiment_slots

influential_feature_list = shap_res_df.feature_name.unique()
feature_counts = shap_res_df.feature_name.value_counts().reset_index().rename(columns={'count':'hit'})
# %%
root_node, node_list = to_tree(LINKAGE_TREES[best_linkage], rd=True)

branch_results = []
for node in node_list:
        if node.is_leaf():
            continue
        branch_leaf_indices = node.pre_order(lambda x: x.id)
        branch_features = X_raw.columns[branch_leaf_indices]
        model_features_in_cluster = list(set(branch_features).intersection(X_raw_final.columns)) 
        hit_features_in_cluster = list(set(branch_features).intersection(influential_feature_list))
        if len(model_features_in_cluster) < 3:
            continue
        if len(hit_features_in_cluster) < 2:
            continue
        # Sum total hits inside this branch using our pre-calculated counts
        k_cluster = int(feature_counts.query('feature_name in @hit_features_in_cluster').hit.sum())
        sample_size_slots = total_cells * len(model_features_in_cluster)
        # Upper-tail Binomial Survival Function: P(X >= k_cluster)
        p_val = binom.sf(k_cluster - 1, sample_size_slots, global_p_baseline)
        branch_results.append(pd.DataFrame({
            'node_id': [node.id],
            'cluster_size': len(branch_features),
            'hit_number':[len(hit_features_in_cluster)],
            'p_value': [p_val]
        }))

df_branches = pd.concat(branch_results)
    
# 7. FDR Correction & Nested-Box Filtering
if not df_branches.empty:
    _, p_adj, _, _ = multipletests(df_branches['p_value'], method='fdr_bh')
    df_branches['fdr_p_value'] = p_adj


# %%
plt_feature= 'Texture_InfoMeas1_Cell-Membrane.CP01_3_01_256'
tmp_feature_name = [plt_feature]
tmp_feature_tbl = X_raw_final.loc[:,tmp_feature_name].assign(label = y_raw)
resistant_cell_tbl = data_tbl.query('treatment == "R1280"').loc[:,tmp_feature_name].assign(label = 'R1280')
tmp_feature_tbl = pd.concat([tmp_feature_tbl,resistant_cell_tbl])
tmp_feature_tbl.columns= ['tmp_feature','label']
tmp_ax = tmp_feature_tbl.groupby('label').tmp_feature.plot.kde(legend=True,title=plt_feature)
plt.show()
feature_counts.feature_name[0]
# %%

corr_matrix = pd.DataFrame(X_raw).corr().abs().to_numpy()
corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
