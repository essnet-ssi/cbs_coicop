import pandas as pd

class ExactStringMatcher:
  def __init__(self, dictionary: dict = None, not_found_value = None) -> None:
    if (dictionary is not None) and (not isinstance(dictionary, dict)):
      print(f"Dictionary must of of type dict, found type: {type(dictionary)}")
      exit(1)

    self.dictionary: dict = dictionary
    self.not_found_value = not_found_value

  def __repr__(self) -> str:
    return f"{type(self).__name__}()"

  def fit(self, X: pd.Series, y: pd.Series):
    if len(X) != len(y):
      print("Unmatching dimensions for X and y: %s and %s." % (len(X), len(y)))
      exit(1)
    
    X = X.astype(str)

    # remove duplicates to have one definition 
    # we keep the last duplicate record
    X = X.drop_duplicates(keep="last")
    y = y.loc[X.index]

    self.dictionary = dict(zip(X, y))

    return self

  def predict(self, X: pd.Series) -> pd.Series:
    if self.dictionary is None:
      print(f"{type(self).__name__} not fitted yet!")
      exit(1)

    y_pred = X.map(lambda x: self.dictionary.get(x, self.not_found_value))

    return y_pred
  

class FuzzyStringMatcher:
  def __init__(self) -> None:
    assert 0, "Not implemented"

  def __repr__(self) -> str:
    pass

  def fit(self, X: pd.Series, y: pd.Series):
    assert 0, "Not implemented"
    return self

  def predict(self, X: pd.Series) -> pd.Series:
    assert 0, "Not implemented"
  
