import pandas as pd

class WriteToFile(object):
  """
  Class to write a dict to a Pandas' dataframe and saved to a .csv
  """

  def __init__(self, load, filename):
    self.dataframe=None
    if(isinstance(load, str)):
      self._load(load)
    self._filename=filename
      
  def _load(self, filepath):
    self.dataframe = pd.read_csv(filepath, index_col[0])

  def _write_to_file(self, filename):
    self.dataframe.to_csv(filename)
    
  def __call__(self, dic):
    if(self.dataframe is None):
      self.dataframe = pd.DataFrame.from_dict(dic)
    else:
      row = pd.DataFrame.from_dict(dic)
      frames = [self.dataframe, row]
      self.dataframe = pd.concat(frames, axis=0, ignore_index=True)
      
    self._write_to_file(self._filename)
