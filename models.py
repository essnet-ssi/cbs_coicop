import os
from collections.abc import Iterable
import pandas as pd
import numpy as np
import multiprocessing

import fasttext
import uuid

CACHE_DIR = os.path.join('.', "__cache")

def preprocess_text(a: pd.Series) -> pd.Series:
  a = a.str.lower()
  a = a.str.replace(r"[^A-z0-9\s]", '', regex=True)
  return a

class FastTextClassifier:
  '''
  The FastTextClassifier is a wrapper class for the FastText
  library for compatibility with scikit-learn API,
  i.e. (fit, predict, predict_proba functions).
  It further performs some of the transformations required 
  for the original FastText implementation.
  '''
  def __init__(self, epoch: int = 5, lr: float = 0.1, loss: str = "softmax", bucket: int = 2_000_000, thread: int = multiprocessing.cpu_count() - 1, preprocess: bool = False, minn: int = 1, maxn: int = 1) -> None:
    self.epoch = epoch
    self.lr = lr
    self.loss = loss
    self.bucket = bucket
    self.loss = loss
    self.thread = thread
    self.preprocess = preprocess
    self.minn = minn
    self.maxn = maxn

    self._model = None

  def __repr__(self) -> str:
    ret = "FastTextClassifier("
    ret += f"epoch={self.epoch}" 
    ret += f", lr={self.lr}" 
    ret += f", loss={self.loss}" 
    ret += f", bucket={self.bucket}" 
    ret += f", loss={self.loss}" 
    ret += f", thread={self.thread}" 
    ret += f", preprocess={self.preprocess}" 
    ret += f", minn={self.minn}" 
    ret += f", maxn={self.maxn}" 
    ret += ")"
    return ret

  def set_params(self, **params) -> None:
    if not all(hasattr(self, attr) for attr in params.keys()):
      print(f"Attribute {attr} not found in class {type(self)}.")
      exit(1)

    for attr, val in params.items():
      setattr(self, attr, val)

  def fit(self, X: pd.Series, y: pd.Series):
    if not os.path.isdir(CACHE_DIR):
      os.mkdir(CACHE_DIR)

    # FastText only accepts .txt files as input.
    # Each instance should be coded as follows in the .txt file:
    #
    # __label__<COICOP_CODE> <RECEIPT_TEXT>
    #
    # For example:
    # __label__011140 beemster kaas
    # 
    # Note that the labels prefix with __label__.
    # We first save X as a .txt file in a cache folder so that FastText
    # can handle the input.

    if self.preprocess: X = preprocess_text(X)

    y_ = y.apply(lambda row: '__label__%s' % row)

    X_y = pd.concat([X, y_], axis=1)
    X_y = X_y.apply(lambda row: ' '.join(row), axis=1)
    X_y = X_y.str.cat(sep='\n')

    cache_fn = f"{str(uuid.uuid4())}.txt"
    cache_path = os.path.join(CACHE_DIR, cache_fn)

    with open(cache_path, 'w') as fp:
      fp.write(X_y)
    
    self._model = fasttext.train_supervised(
      cache_path,
      epoch=self.epoch,
      lr=self.lr,
      loss=self.loss,
      bucket=self.bucket,
      thread=self.thread,
      minn=self.minn,
      maxn=self.maxn,
    )
    return self

  def predict(self, X: pd.Series) -> np.ndarray:
    if self._model is None:
      print("ERROR: Model not fitted yet.")
      exit(1)
    
    if self.preprocess: X = preprocess_text(X)

    X = X.tolist() # Convert to list so that FastText can handle it

    y_pred, _ = self._model.predict(X)

    y_pred = np.array(y_pred)
    # remove __label__ that was prepended by FastText
    y_pred = np.char.replace(y_pred, "__label__", '') 
    y_pred = y_pred.flatten()
    return y_pred

#  def predict_proba(self, X: pd.Series) -> np.ndarray:
#    print("Not implemented yet!")
#    exit(1)
#    pass



