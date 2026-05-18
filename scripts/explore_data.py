%autoindent off
# %%
import pandas as pd
from matplotlib import pyplot as plt
# %%
data_tbl = pd.read_csv('~/Documents/Teton_project/data/morphology_metadata_TETON.csv')

# %%
columns_select = []
columns_select.extend(['Cell'])
columns_select.extend(list(data_tbl.columns[11:601]))

# %%
morph_data_tbl = data_tbl.loc[:,columns_select] 

# %%

tmp_ax = morph_data_tbl.Area.plot.kde()
plt.show()
