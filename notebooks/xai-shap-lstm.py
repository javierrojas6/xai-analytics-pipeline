# %%
import os
import sys

import numpy as np
import pandas as pd
import shap
import torch
import torchinfo
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

#%% 
sys.path.append(os.path.join(os.getcwd(), '../src'))
from app.model.ctl_classifier_lstm_v1 import build_lstm_v1

# %%
sequence_length = 68
category = ['UNDECIDED', 'VERY_BAD', 'VERY_GOOD']
dataset_filename = '../data/model-1/dataset-v2.csv'
model_path = '../trained/lsmt-v1-20231013151005.cpu.pth'
Y_FIELD = 'rating'
# %%
model = build_lstm_v1()
model.load_state_dict(torch.load(model_path))
model.eval()

# %%
model

# %%
sequence = [317, 125, 843, 304]
sequence = torch.tensor(sequence, dtype=torch.float32)
sequence = pad_sequence([sequence, torch.zeros(sequence_length)], padding_value=0, batch_first=True)
sequence = torch.reshape(sequence[0], (1, 1, sequence_length))
# %%
prediction = model(sequence)
predicted_rating = np.argmax(prediction.detach().numpy())
print(category[int(predicted_rating)])
# %%
torchinfo.summary(model, (1, 1, 68))
# %%
df = pd.read_csv(dataset_filename, sep=',', index_col=0)
# %%
df.info()

#%% 
class MyDataset(Dataset):
    y_encoder = None
    x_encoder = None

    def __init__(self, dataset, y_label: str, standardize: bool = False):
        sequences = []
        for _, row in dataset.iterrows():
            row_sequence = row['sequence'].split('|')
            row_sequence = list(map(lambda x: 0 if x == -1 else x, map(int, row_sequence)))

            sequences += [row_sequence]

        if standardize:
            x_encoder = preprocessing.StandardScaler()
            x_encoder.fit(sequences)

            sequences = x_encoder.transform(sequences)
            
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(sequences)
            sequences = scaler.transform(sequences)

        self.x = torch.tensor(sequences, dtype=torch.float)
        dim = (self.x.size(0), 1, self.x.size(1))
        self.x = torch.reshape(self.x, dim)

        self.y_encoder = preprocessing.LabelEncoder()

        y = self.y_encoder.fit_transform(dataset[y_label].values)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_dataframe(self) -> pd.DataFrame:
        x_array = torch.reshape(self.x, (self.x.size(0), self.x.size(-1))).detach().numpy()
        tmp = list(map(lambda x: int(x), list(self.y.detach().numpy())))
        # y_array = self.y_encoder.inverse_transform(tmp)
        y_array = tmp
        
        df = pd.DataFrame(data=x_array)
        df[df.shape[1]] = y_array
        return df
# %%
ds = MyDataset(df, Y_FIELD, True)
# %%
df_s = ds.get_dataframe()
# %%
df_s
# %%
train_dataset, test_dataset = train_test_split(df_s, test_size=0.3)

# %%
explainer = shap.DeepExplainer(model, ds[:100])