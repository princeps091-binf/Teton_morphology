%autoindent off
# %%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
# %%
data_tbl = pd.read_csv('~/Documents/Teton_project/data/morphology_metadata_TETON.csv')

# %%
# From basic column name examination the indices below seem to recover all the morphological feature and exclude the sequencing features
columns_select = []
columns_select.extend(['Cell'])
columns_select.extend(list(data_tbl.columns[11:601]))

# %%
morph_data_tbl = data_tbl.loc[:,columns_select] 

# %%
morph_cor_mat = morph_data_tbl.iloc[:,5:-9].corr().to_numpy()

# %%
distance_matrix = np.clip(1 - np.abs(morph_cor_mat), 0, 1)
condensed_distance = squareform(distance_matrix)
linkage = hierarchy.average(condensed_distance)
cluster_ids = hierarchy.fcluster(linkage, t=0.6, criterion='distance')

# Map features to their respective clusters
feature_clusters = pd.DataFrame({
    'feature':  morph_data_tbl.iloc[:,5:-9].columns,
    'cluster_id': cluster_ids
})
reordered_indices = hierarchy.leaves_list(linkage)
max(cluster_ids)
# Reorder the correlation matrix symmetrically
reordered_corr = morph_cor_mat[reordered_indices, :][:, reordered_indices]
reordered_clusters = cluster_ids[reordered_indices]

# %%
from matplotlib.colors import ListedColormap
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
