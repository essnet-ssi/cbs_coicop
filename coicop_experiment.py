import os
import csv
from hashlib import sha256
import pandas as pd
import numpy as np
import pickle as pkl
from datetime import datetime

from sklearn.base import clone
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.model_selection import PredefinedSplit, GridSearchCV

import config
from hierarchical.metrics import hierarchical_precision_score, hierarchical_recall_score, hierarchical_f1_score

def sample_train_data(df: pd.DataFrame, sample_strategy: str | None) -> pd.DataFrame:
  if sample_strategy is None:
    return df
  elif sample_strategy == "stratify_store":
    n_stratification_samples = 20_000
    return df.groupby("store_name").sample(n=n_stratification_samples, replace=False, random_state=config.SEED)
  elif sample_strategy == "random_sample":
    n_samples = min(len(df), 100_000)
    return df.sample(n=n_samples, replace=False, random_state=config.SEED)
  else:
    print(f"Unknown sampling strategy: {self.train_sample_strategy}")
    exit(1)

def score(y_test: np.ndarray, y_pred: np.ndarray, sample_weights: np.ndarray = None) -> dict[str, float | None]:
  exp_results: dict[str, float | None] = dict()

  exp_results["accuracy"] = accuracy_score(y_test, y_pred, sample_weight=sample_weights)
  exp_results["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred, sample_weight=sample_weights)
  exp_results["precision"] = precision_score(y_test, y_pred, average="macro", zero_division=1, sample_weight=sample_weights)
  exp_results["recall"] = recall_score(y_test, y_pred, average="macro", zero_division=1, sample_weight=sample_weights)
  exp_results["f1"] = f1_score(y_test, y_pred, average="macro", zero_division=1, sample_weight=sample_weights)
  exp_results["log_loss"] = None

  return exp_results

def hash_pipeline(pipeline: Pipeline, param_grid: dict, X_train: pd.DataFrame, y_train: pd.Series):
  #_ make an identifier for the pipeline by hashing the training set and pipeline steps
  # this id is used to load and save the fitted pipeline
  pipeline_hash = sha256(repr(pipeline).encode("ascii")).hexdigest()
  param_grid_hash = sha256(repr(param_grid).encode("ascii")).hexdigest()
  X_train_hash  = sha256(pd.util.hash_pandas_object(X_train).values).hexdigest()
  y_train_hash  = sha256(pd.util.hash_pandas_object(y_train).values).hexdigest()

  pipeline_hash = sha256(f"{X_train_hash}{y_train_hash}{pipeline_hash}{param_grid_hash}".encode("ascii")).hexdigest()
  return pipeline_hash
      
def write_csv(out: dict[str, float], output_path: str):
  out_exists = os.path.isfile(output_path)

  with open(output_path, "a+") as fp:
    writer = csv.DictWriter(fp, delimiter=';', fieldnames=out.keys())

    if not out_exists:
      writer.writeheader()

    writer.writerow(out)

class CoicopExperiment:
  def __init__(self, pipeline: Pipeline, param_grid: dict, train_sample_strategy: str | None = None) -> None:
    self.pipeline = pipeline 
    self.param_grid = param_grid
    self.train_sample_strategy = train_sample_strategy

    self.metrics = [
      "accuracy",
      "balanced_accuracy",
      "precision",
      "recall",
      "f1",
      "log_loss",
      "hierarchical_precision",
      "hierarchical_recall",
      "hierarchical_f1",
    ]


  def evaluate(
    self,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    val_idx ,
    test_label: str,
    X_col_names: str | list[str],
    y_col_name: str,
    predict_level: int,
    results_fn: str,
    weights_col_name: str | None = None,
    hierarchical_split_func: callable = None,
    experiment_annotations: str = None,
    save_pipeline: bool = True
  ) -> None:
    
    # get X, y's train
    df_train = sample_train_data(df_train, self.train_sample_strategy)

    X_train = df_train[X_col_names]
    y_train = df_train[y_col_name]

    pipeline_fn = "%s.pkl" % hash_pipeline(self.pipeline, self.param_grid, X_train, y_train)
    pipeline_path = os.path.join(config.PIPELINE_DIR, pipeline_fn)

    # fit or load model
    # -----------------
    if os.path.isfile(pipeline_path):
      # Load pipeline
      fp = open(pipeline_path, "rb")
      pipeline = pkl.load(fp)
      fp.close()
      
    else:
      # fit and save best pipeline from grid search
      # refactor predefined split out
      grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=val_idx, scoring="accuracy", refit=True, n_jobs=-1)
      grid_search.fit(X_train, y_train)

      pipeline = grid_search.best_estimator_

      fp = open(pipeline_path, "wb")
      pkl.dump(pipeline, fp)
      fp.close()

    # get X, y test
    X_test = df_test[X_col_names]
    y_test = df_test[y_col_name]
    sample_weights = df_test[weights_col_name] if weights_col_name is not None else None

    #
    # check if labels in test set occur in training set
    #
    train_labels = set(y_train)
    test_labels  = set(y_test)

    # calculate scores
    # ----------------
    pipeline_name = [str(step) for step in pipeline.named_steps.values()]
    pipeline_name = ', '.join(pipeline_name)

    metadata = {
      "pipeline": pipeline_name,
      "predict_level": predict_level,
      "fit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      "test_label": test_label,
      "experiment_annotations": experiment_annotations,
    }

    y_pred = pipeline.predict(X_test)

    exp_results = score(y_test, y_pred, sample_weights)

    # additional metrics
    # @todo, refactor properly into the score function
    if hasattr(pipeline, "predict_proba"):
      y_proba = pipeline.predict_proba(X_test)

      np.nan_to_num(y_proba, copy=False, nan=0) # sometimes y_proba yields nans 
      exp_results["log_loss"] = log_loss(y_test, y_proba, labels=y_train)

    y_pred = pd.Series(y_pred)
    y_test = y_test.reset_index(drop=True)

#      if hierarchical_split_func is not None:
#        exp_results["hierarchical_precision"] = hierarchical_precision_score(y_test, y_pred, hierarchical_split_func, predict_level, average="macro")
#        exp_results["hierarchical_recall"] = hierarchical_recall_score(y_test, y_pred, hierarchical_split_func, predict_level, average="macro")
#        exp_results["hierarchical_f1"] = hierarchical_f1_score(y_test, y_pred, hierarchical_split_func, predict_level, average="macro")

    assert all(exp_metric in self.metrics for exp_metric in exp_results), f"Found key(s) not declared in self.results. Keys in exp_results: {exp_results.keys()}"

    # write to disk
    # --------------
    out = metadata | exp_results
    output_path = os.path.join(config.RESULTS_DIR, results_fn)

    write_csv(out, output_path)


