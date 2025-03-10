import numpy as np
import pandas as pd

def perfect_comparability_percentage(y_true: pd.Series, y_pred: pd.Series):
  pass


def comparability_ratio(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
  if isinstance(y_true, np.ndarray): y_true = pd.Series(y_true)
  if isinstance(y_pred, np.ndarray): y_pred = pd.Series(y_pred)

  assert len(y_true) == len(y_pred), f"Unmatching dimensions in y_true and y_pred {(len(y_true), len(y_pred))}" 

  y_true_counts = y_true.value_counts()
  y_pred_counts = y_pred.value_counts()

  # add missing keys in both sides
  y_labels = set(y_true)
  y_labels.update(y_pred)

  # add missing labels
  y_true_counts = y_true_counts.reindex(y_labels)
  y_pred_counts = y_pred_counts.reindex(y_labels)

  y_true_counts = y_true_counts.sort_index()
  y_pred_counts = y_pred_counts.sort_index()

  y_true_counts = pd.Series.div(y_true_counts, len(y_true), fill_value=None)
  y_pred_counts = pd.Series.div(y_pred_counts, len(y_true), fill_value=None)

  ret = pd.Series.div(y_pred_counts, y_true_counts, fill_value=None)
  ret = ret.sort_index()
  ret = ret.fillna(0)
  return ret

  
def bias(y_true: pd.Series, y_pred: pd.Series, absolute: bool = False) -> int:
  if not isinstance(y_true, pd.Series): y_true = pd.Series(y_true)
  if not isinstance(y_pred, pd.Series): y_pred = pd.Series(y_pred)

  y_true_counts = y_true.value_counts()
  y_pred_counts = y_pred.value_counts()

  ret = pd.Series.sub(y_pred_counts, y_true_counts)
  if absolute: ret = ret.abs()
  ret = ret.fillna(0)
  ret = ret.sum()
  return ret

