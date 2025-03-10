import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy

import config

class LabseExtractor:
  def __init__(self, model_path: str | None = None) -> None:
    if model_path is None:
      model_path = config.FEATURE_EXTRACTION_LABSE_PATH

    self.extractor = SentenceTransformer(model_name_or_path=model_path, local_files_only=True)

  def __repr__(self):
    return "LabseExtractor()"

  def fit(self, X: pd.Series | None = None):
    return self

  def transform(self, X: pd.DataFrame | pd.Series, y: pd.Series | None = None) -> np.ndarray:
    X_ = X.reset_index(drop=True) # unfortunately needed due to a key error bug in the LabseModel
    return self.extractor.encode(X_)

  def fit_transform(self, X: pd.DataFrame | pd.Series, y: pd.Series | None = None) -> np.ndarray:
    return self.transform(X, y)


class SpacyExtractor:
  def __init__(self, model_name: str, n_jobs=1) -> None:
    supported_models = ("nl_core_news_sm", "nl_core_news_md", "nl_core_news_lg")

    if model_name not in supported_models:
      print(f"Unsupported model_name: {model_name}")
      exit(1)

    self.extractor = spacy.load(model_name)
    self.model_name = model_name
    self.n_jobs = n_jobs

  def __repr__(self):
    return f"SpacyExtractor(model_name={self.model_name}, n_jobs={self.n_jobs})"

  def fit(self, X: pd.Series | None = None):
    return self

  def transform(self, X: pd.Series, y: pd.Series | None = None) -> np.ndarray:
    docs = self.extractor.pipe(X, n_process=self.n_jobs, disable=["parser", "tagger", "ner"])
    ret = [doc.vector for doc in docs]
    ret = np.array(ret)
    return ret


  def fit_transform(self, X: pd.Series, y: pd.Series | None = None) -> np.ndarray:
    return self.transform(X, y)

