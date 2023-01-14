import pandas as pd
import yaml
from yaml.loader import SafeLoader

class CarsData:
  def get_dataloader(self):
    self.df = pd.read_csv('.\Data\Cars\Cars.csv').drop(['Unnamed: 0'], axis = 1)
    self.keys = self.df.keys()
    self.row_list = []
    self.normalize_vals = []
    #print(self.df)
    for col_name in self.keys:
      if self.df[col_name].dtype == 'object':
        if col_name == 'Mh':
          with open('.\\resources\car_manufacturrers.yml') as f:
            data = yaml.load(f, Loader=SafeLoader)
            data = [x.upper() for x in data]
            n = len(col_vals)
            vals = list(range(1, n+1))
            di = dict(zip(data, vals))
            #print(di)
            new_col = []
            for el in data:
              for idx, val in enumerate(self.df[col_name]):
                if el in val:
                  #self.df[col_name][idx] = el
                  new_col.append(el)
            self.df[col_name] = new_col
            self.df[col_name] = self.df[col_name].map(di).fillna(0)
        else: 
          col_vals = list(self.df[col_name].unique())
          n = len(col_vals)
          vals = list(range(1, n+1))
          di = dict(zip(col_vals, vals))
          self.df[col_name] = self.df[col_name].map(di).fillna(0)
    self.df = self.df.fillna(0)
    self.norm_mean_std = []
    for col_name in self.keys:
      m = self.df[col_name].max()
      if m != 0:
        self.df[col_name] = self.df[col_name] / m 
      self.norm_mean_std.append(m)

    print(self.df)

    for _, row in self.df.iterrows():
      self.row_list.append(row.to_list())
    return self.row_list

d = CarsData()
d.get_dataloader()