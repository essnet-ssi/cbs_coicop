import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import config

if __name__ == "__main__":
  dataset_fn = "lidl_ah_jumbo_plus.parquet"
  dataset_path = os.path.join(config.OUTPUT_DATA_DIR, dataset_fn)

  df_stores = pd.read_parquet(dataset_path)

  # set up output directory for graphics
  if not os.path.isdir(config.OUTPUT_GRAPHICS_DIR):
    os.mkdir(config.OUTPUT_GRAPHICS_DIR)

  #
  # Number of records per month for each supermarket
  # x-axis: months, y-axis stores
  #
  x_axis_col = "year_month"
  y_axis_col = "store_name"
  val_col    = "count"

  graph_data = df_stores.groupby([x_axis_col, y_axis_col])[val_col].sum()
  graph_data = graph_data.reset_index()
  graph_data = graph_data.pivot(index=x_axis_col, columns=y_axis_col)
  graph_data = graph_data.transpose() # flip x and y such that year_month is displayed at the bottom

  sns.heatmap(graph_data, cmap="Blues", robust=True)

  out_fn = "stores__inventory_per_month.png"
  plt.savefig(os.path.join(config.OUTPUT_GRAPHICS_DIR, out_fn))

  # graphic for common period in which we have data for all stores
  graph_data_common = graph_data.dropna(axis=1, how="any")

  plt.clf()
  sns.heatmap(graph_data_common, cmap="Blues", robust=True)

  out_fn = "stores__inventory_per_month_common.png"
  plt.savefig(os.path.join(config.OUTPUT_GRAPHICS_DIR, out_fn))

  # graphic for cells greater than 0
  graph_data_bin = graph_data > 0

  plt.clf()
  sns.heatmap(graph_data_bin, cmap="Blues")

  out_fn = "stores__inventory_per_month_bin.png"
  plt.savefig(os.path.join(config.OUTPUT_GRAPHICS_DIR, out_fn))

  #
  # plot svd
  #

  from sklearn.feature_extraction.text import HashingVectorizer
  from sklearn.decomposition import TruncatedSVD
  import seaborn as sns

  text_col = "receipt_text"
  #text_col = "ean_name"
  coicop_level = 5

  hv = HashingVectorizer(input="content", binary=True, norm="l2", analyzer="char", ngram_range=(3, 6), n_features=100_000)

  df_stores_sample = df_stores.drop_duplicates(subset=text_col, keep="first")
  #df_stores_sample = df_stores_sample[df_stores_sample[f"coicop_level_{coicop_level - 1}"] == "01"]
  df_stores_sample = df_stores_sample.sample(frac=0.1)
  df_stores_sample = df_stores_sample.sort_values(by="coicop_number")

  texts = df_stores_sample[text_col]
  #coicop_labels = df_stores_sample[f"coicop_level_{coicop_level}"]
  coicop_labels = df_stores_sample[f"coicop_number"]

  text_count_matrix = hv.fit_transform(texts)
  svd = TruncatedSVD(n_components=2)

  svd.fit(text_count_matrix)
  text_svd = svd.transform(text_count_matrix)

  plt.clf()
  out_fn = "receipt_text__dimension_reduced.png"
  sns.scatterplot(x=text_svd[:, 0], y=text_svd[:, 1], hue=coicop_labels)
  plt.savefig(os.path.join(config.OUTPUT_GRAPHICS_DIR, out_fn))

