import pandas as pd

class CarsData:
  def get_dataloader(self):
    self.df = pd.read_csv('.\Data\Cars.csv')
    self.keys = self.df.keys()
    self.row_list = []
    for _, row in self.df.iterrows():
      self.row_list.append(row.to_list())
    return self.row_list

