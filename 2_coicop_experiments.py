import os
import copy
from itertools import permutations

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import PredefinedSplit, LeaveOneGroupOut

# local imports
import config
from coicop_experiment import CoicopExperiment
import experiment_parameters as exp_params


def get_coicop_level_label(y: pd.Series, level: int, to_numpy=False) -> pd.Series | np.ndarray:
  if not isinstance(y, pd.Series): y = pd.Series(y)
     
  if not (0 < level and level <= 5):
    print("Undefined COICOP level!")
    exit(1)

  label_pos_stop = level + 1
  ret = y.str.slice(start=0, stop=label_pos_stop)
  if to_numpy: ret = ret.to_numpy()
  return ret

def main(experiment_type: str) -> None:
  experiments: list[CoicopExperiment] = exp_params.EXPERIMENTS
  predict_level: int = exp_params.PREDICT_LEVEL
  experiment_annotations: str = exp_params.EXPERIMENT_ANNOTATIONS
  results_fn: str = exp_params.RESULTS_FN
  save_pipeline: bool = exp_params.SAVE_PIPELINE
  split_tests_by: str | None = exp_params.SPLIT_TESTS_BY
  weights_col_name: str | None = exp_params.WEIGHTS_COL_NAME
  include_999999 = exp_params.INCLUDE_999999

  X_col_name = "receipt_text"
  y_col_name = f"coicop_level_{predict_level}"

  #
  # get train test sets
  #
  train_test_sets: dict[str, tuple[pd.DataFrame, pd.DataFrame, list]] = dict() # last is validation indices so that grid search can work with it. can be a cross validation object.

  if experiment_type == "time":
    test_periods = ["202306", "202307", "202308"]

    df_train_fn = "train_lidl_ah_jumbo_plus.parquet"
    df_train_path = os.path.join(config.OUTPUT_DATA_DIR, df_train_fn)
    
    df_train = pd.read_parquet(df_train_path)

    for test_period in test_periods:
      df_test_fn   = f"test_{test_period}_lidl_ah_jumbo_plus_incl_999999.parquet"
      df_test_path = os.path.join(config.OUTPUT_DATA_DIR, df_test_fn)

      df_test = pd.read_parquet(df_test_path)
      # remove later
      df_test = df_test[df_test["receipt_text"].notna()]

      if not include_999999: df_test = df_test[df_test["coicop_level_5"] != "999999"]

      # get validation indices
      val_month = max(df_train["year_month"])
      val_idx = PredefinedSplit(df_train["year_month"] == val_month)

      if not split_tests_by:
        train_test_sets[test_period] = (df_train, df_test, val_idx)
      else:
        for store_name, df_test in df_test.groupby(split_tests_by):
          train_test_sets[f"{test_period}_{store_name}"] = (df_train, df_test, val_idx)

  elif experiment_type == "store":
    if split_tests_by: print("INFO: Split_test_by is set to non-null during a store experiment. Tests will not be split.")

    df_full_fn = "full_lidl_ah_jumbo_plus_incl_999999.parquet"
    df_full_path = os.path.join(config.OUTPUT_DATA_DIR, df_full_fn)

    df_full = pd.read_parquet(df_full_path)

    store_col_name = "store_name"

    # drop jumbo due to errors in data linkage
    print("INFO: dropping instances from Jumbo.")
    df_full = df_full[df_full[store_col_name] != "jumbo"]
    unique_stores = df_full[store_col_name].unique()

    for test_store in unique_stores:
      df_train = df_full[df_full[store_col_name] != test_store]
      df_test  = df_full[df_full[store_col_name] == test_store]

      df_train = df_train[df_train["coicop_level_5"] != "999999"]
      if not include_999999: df_test = df_test[df_test["coicop_level_5"] != "999999"]

      val_idx = LeaveOneGroupOut().split(df_train, groups=df_train["store_name"])

      train_test_sets[test_store] = (df_train, df_test, val_idx)

  else:
    print(f"Unknown experiment type: {experiment_type}")
    exit(1)

  assert len(train_test_sets) > 0, "No train and test sets to evaluate on."

  #
  # evaluate the train test evaluations
  #
  while len(experiments) > 0:
    exp = experiments.pop()

    for test_label, (df_train, df_test, val_idx) in train_test_sets.items():
      exp.evaluate(
        df_train,
        df_test,
        val_idx,
        test_label=test_label,
        X_col_names=X_col_name,
        y_col_name=y_col_name,
        predict_level=predict_level,
        weights_col_name=weights_col_name,
        results_fn=results_fn,
        hierarchical_split_func=get_coicop_level_label,
        experiment_annotations=experiment_annotations,
        save_pipeline=save_pipeline,
      )

def setup_dir(dir_path: str) -> None:
  if not os.path.isdir(dir_path):
    os.mkdir(dir_path)

if __name__ == "__main__":
  experiment_type = exp_params.EXPERIMENT_TYPE

  # setup output directories
  setup_dir(config.OUTPUT_GRAPHICS_DIR)
  setup_dir(config.RESULTS_DIR)
  setup_dir(config.PIPELINE_DIR)

  main(experiment_type)

