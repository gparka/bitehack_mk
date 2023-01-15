import pandas as pd
import yaml
from yaml.loader import SafeLoader
import torch
from torchvision import transforms
import numpy as   np
from torch.utils.data import Dataset, DataLoader

class CarsData:
  def get_data(self, download = False, save = False):
    if download:
      path = "http://co2cars.apps.eea.europa.eu/tools/download?download_query=http%3A%2F%2Fco2cars.apps.eea.europa.eu%2F%3Fsource%3D%7B%22track_total_hits%22%3Atrue%2C%22query%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22constant_score%22%3A%7B%22filter%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22bool%22%3A%7B%22should%22%3A%5B%7B%22term%22%3A%7B%22year%22%3A2021%7D%7D%5D%7D%7D%2C%7B%22bool%22%3A%7B%22should%22%3A%5B%7B%22term%22%3A%7B%22scStatus%22%3A%22Provisional%22%7D%7D%5D%7D%7D%5D%7D%7D%7D%7D%5D%7D%7D%2C%22display_type%22%3A%22tabular%22%7D&download_format=csv"
    else:
      path = '.\..\Datasets\Cars.csv'
    self.df = pd.read_csv(path).drop(['ID', 'VFN', 'Mp', 'Man',
              'MMS', 'T', 'Va', 'Ve', 'Cn', 'Ct', 'Cr', 'At1 (mm)', 'At2 (mm)', 
              'IT', 'Vf', 'year'], axis = 1)
    self.keys = self.df.keys()
    self.row_list = []
    self.normalize_vals = []
    for col_name in self.keys:
      if self.df[col_name].dtype == 'object':
        if col_name == 'Mk':
          with open('.\\resources\car_manufacturers.yml') as f:
            data = yaml.load(f, Loader=SafeLoader)
            data = [x.upper() for x in data]
            n = len(col_vals)
            vals = list(range(1, n+1))
            di = dict(zip(data, vals))
            self.df[col_name] = self.df[col_name].map(di).fillna(0)
        else: 
          col_vals = list(self.df[col_name].unique())
          n = len(col_vals)
          vals = list(range(1, n+1))
          di = dict(zip(col_vals, vals))
          self.df[col_name] = self.df[col_name].map(di).fillna(0)
    self.df = self.df.fillna(0)

    for _, row in self.df.iterrows():
      self.row_list.append(row.to_list())

    if save:
      self.df.to_csv('.\\resources\cleanData.csv')

    return self.row_list

class CarDataset(Dataset):
  def __init__(self):
    self.df = pd.read_csv('.\\resources\cleanData.csv')
    self.keys = self.df.keys()
    self.norm_reverse = []
    for col_name in self.keys:
      m = self.df[col_name].max()
      if m != 0:
        self.df[col_name] = self.df[col_name] / m 
      self.norm_reverse.append(m)
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    data = self.df.iloc[idx, :]
    data = np.asarray(data)
    data = torch.Tensor(data)
    return data

def get_dataloader(data, batch_s, shuff = True, n_workers = 0):
  return DataLoader(data, batch_size = batch_s, shuffle = shuff, num_workers = n_workers)

if __name__ == "__main__":
    cars = CarsData()
    cars.get_data()