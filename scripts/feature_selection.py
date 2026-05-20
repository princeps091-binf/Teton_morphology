%autoindent off
# %%
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ParameterGrid
from sklearn.feature_selection import mutual_info_classif
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# %%
class ClusterFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, linkage_method='ward', criterion='distance', t=0.3, feature_correlation_matrix=None, feature_mi_score = None):
        self.linkage_method = linkage_method
        self.criterion = criterion
        self.t = t
        self.corr = feature_correlation_matrix
        self.mi_scores = feature_mi_score
        self.selected_features_ = None
        self.aggregate_mi_ = 0.0
        self.average_mi_ = 0.0
        self.feature_clusters = None

    def fit(self, X, y):
        X_df = pd.DataFrame(X)
        
        # 1. Calculate MI scores for ALL features upfront
        if self.mi_scores is None:
            self.mi_scores = mutual_info_classif(X_df, y, random_state=42)
        
        # 2. Compute the distance matrix
        if self.corr is None:
            corr = X_df.corr().abs().to_numpy()
            corr = np.nan_to_num(corr, nan=0.0)
        distance_matrix = np.clip(1 - self.corr, 0, 1)
        
        # 3. Perform hierarchical clustering
        condensed_dist = squareform(distance_matrix)
        
        # Handle the specific linkage requested by the grid
        if self.linkage_method == 'ward':
            linkage = hierarchy.ward(condensed_dist)
        elif self.linkage_method == 'average':
            linkage = hierarchy.average(condensed_dist)
        elif self.linkage_method == 'complete':
            linkage = hierarchy.complete(condensed_dist)
            
        # 4. Generate cluster assignments
        try:
            cluster_ids = hierarchy.fcluster(linkage, t=self.t, criterion=self.criterion)
        except ValueError:
            # If a t/criterion combination is mathematically invalid, fallback gracefully
            self.selected_features_ = list(X_df.columns)
            return self

        # 5. Extract the best representative per cluster
        self.feature_clusters = pd.DataFrame({
            'feature_idx': range(X_df.shape[1]),
            'cluster_id': cluster_ids,
            'mi_score': self.mi_scores
        })
        
        best_features = (
            self.feature_clusters
            .sort_values(by='mi_score', ascending=False)
            .groupby('cluster_id')
            .first()
        )
        
        self.selected_features_ = best_features['feature_idx'].tolist()
        
        # Calculate evaluation metrics
        self.aggregate_mi_ = best_features['mi_score'].sum()
        self.average_mi_ = best_features['mi_score'].mean() if len(self.selected_features_) > 0 else 0
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df.iloc[:, self.selected_features_]

# %%

data_tbl = pd.read_csv('~/Documents/Teton_project/data/morphology_metadata_TETON.csv')
treatment_labels = ['Ctrl','40nMImatinib']

modeling_data_tbl = data_tbl.query('treatment in @treatment_labels')


# %%

columns_select = []
#columns_select.extend(['Cell'])
columns_select.extend(list(data_tbl.columns[11:592]))



X_raw = modeling_data_tbl.loc[:,columns_select] 
y_raw = modeling_data_tbl.treatment.to_numpy()

# %%
# compute the correlation matrix and the mutual information criteria before the grid search

morph_cor_mat = X_raw.corr().to_numpy()
morph_feature_mi_score = mutual_info_classif(X_raw, y_raw, random_state=42)

# %%
# Define the search space
# Note: thresholds mean completely different things for different setups!
param_grid = [
        {'linkage_method': ['average'], 'criterion': ['distance'], 't': list(np.linspace(0,1,21)), 'feature_correlation_matrix':[morph_cor_mat],'feature_mi_score':[morph_feature_mi_score]}
]

best_score = -1
best_params = None

for params in ParameterGrid(param_grid):
    selector = ClusterFeatureSelector(**params)
    selector.fit(X_raw, y_raw)
    # CRUCIAL: We optimize for AVERAGE MI per feature to penalize massive, redundant sets.
    # Alternatively, use a custom composite score like: selector.aggregate_mi_ - (0.05 * len(selector.selected_features_))
    current_score = selector.average_mi_ 
    print(selector.feature_clusters.cluster_id.max()) 
    if current_score > best_score:
        best_score = current_score
        best_params = params
        best_feature_count = len(selector.selected_features_)


