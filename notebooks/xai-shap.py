# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# %%
data = pd.read_csv('../data/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
# model =  LogisticRegression(random_state=0).fit(train_X, train_y)

# %%
X.shape
# %%
row_to_show = 5
# %%
data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

# %%
model.predict_proba(data_for_prediction_array)

# %%
import shap  # package used to calculate Shap values
# %%
shap.initjs()

# %%
print('RandomForestClassifier')

# %%
# Create object that can calculate shap values
# explainer = shap.TreeExplainer(model)
explainer = shap.Explainer(model)
# %%
explainer
# %%
# Calculate Shap values
# shap_values = explainer(X)
shap_values = explainer(X)

# %%
shap_values
# %%
rows_n, attr_n, vals_n = shap_values.shape
print('dim: ', rows_n, attr_n, vals_n )
shap_values_row = shap_values[row_to_show, :, :]
# %%
shap_values_row.values
# %%
explainer.expected_value
# %%
shap.plots.waterfall(shap_values[5, :, 0])
# %%
shap.plots.bar(shap_values[5, :, 0])
# %%
shap.plots.beeswarm(shap_values[:, :, 1])

# %%
# use Kernel SHAP to explain test set predictions
k_explainer = shap.KernelExplainer(model.predict_proba, train_X)
k_shap_values = k_explainer.shap_values(data_for_prediction)
shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], data_for_prediction)

# %%
shap.plots.force(shap_values[5, :, 0])
# %%
shap.plots.beeswarm(shap_values[:, :, 1])

## ##