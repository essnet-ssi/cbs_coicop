import os
import pandas as pd 

import config

# columns
# 'store_id', 'year_month', 'ean_name', 'year', 'month',
# 'revenue', 'amount',  
# 'rep_id', 'product_id',
# 'isba_number', 'isba_name', 'esba_number', 'esba_name',
# 'coicop_name', 'coicop_number', 'coicop_level_1', 'coicop_level_2', 'coicop_level_3', 'coicop_level_4', 'store_name',
# 'receipt_text'


LOAD_COLUMNS = [
 'ean_name',
 'product_id',
 'receipt_text', 
 'store_id',
 'store_name',
 'coicop_name',
 'coicop_number',
 'coicop_level_1',
 'coicop_level_2',
 'coicop_level_3',
 'coicop_level_4',
 'year_month',
 'year',
 'month',
 'revenue',
 'amount',
]

COL_NAME_MAPPING = {
  "coicop_number": "coicop_level_5"
}

def preprocess(df: pd.DataFrame, assign_weights=False, group_by_period=True, filter_records=True, drop_999999=True) -> pd.DataFrame:
  df = df[df["receipt_text"].notna()]
  df["store_name"] = df["store_name"].replace(to_replace={"ah_franchise": "ah"}) # merge ah_franchise (i.e. ah-to-go) which ah
  df = df.sort_values("year_month", ascending=False)

  if filter_records:
    df = df[df["receipt_text"].notna()]
    df = df[df["receipt_text"].str.len() > 1]

  if drop_999999:
    df = df[df["coicop_level_5"] != "999999"] # drop 99999 (ununsed category)

  if group_by_period:
    # 
    # remove duplicate records based on groupby_cols
    # 
    groupby_cols = ["ean_name", "receipt_text", "coicop_level_5"] # group by ean name, receipt text, and coicop

    assert all(col in df.columns for col in groupby_cols)

    # sum the "amount" of duplicate records (by groupby_cols), this is the "number sold"
    df = df.set_index(groupby_cols)
    df["number_sold"] = df.groupby(groupby_cols)["amount"].sum()
    df["revenue"] = df.groupby(groupby_cols)["revenue"].sum()
    df = df.reset_index()

    df = df.drop("amount", axis=1)

    df = df.drop_duplicates(subset=groupby_cols, keep="first") # sorted by date, most recent first, so keep only the newest entry

  return df

def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[pd.DataFrame]]:
  # take 2023-05 and 2023-06 as test set
  periods_in_test_set: list[str] = ["202306", "202307", "202308"]
  max_period_train_set: str = min(periods_in_test_set)

  df_train = df[df["year_month"] < max_period_train_set]
  df_tests: dict[pd.DataFrame] = {period: df[df["year_month"] == period] for period in periods_in_test_set}
    
  return df_train, df_tests

def write_dataset(df: pd.DataFrame, out_fn: str, write_metadata=True) -> None:
  if not os.path.isdir(config.OUTPUT_DATA_DIR):
    os.mkdir(config.OUTPUT_DATA_DIR)

  output_path = os.path.join(config.OUTPUT_DATA_DIR, out_fn)
  df.to_parquet(output_path)

  if write_metadata:
    metadata_out_fn, _ = os.path.splitext(out_fn)
    metadata_out_fn = f"{metadata_out_fn}_metadata.txt"
    metadata_output_path = os.path.join(config.OUTPUT_DATA_DIR, metadata_out_fn)

    out = (
      "==================================\n"
      f"{out_fn}\n"
      "===================================\n\n"
    )
    
    out += f"Num. of Rows   : {df.shape[0]}\n"
    out += f"Num. of Columns: {df.shape[1]}\n"
    out += f"Min. Period    : {df['year_month'].min()}\n"
    out += f"Max. Period    : {df['year_month'].max()}\n"
    out += f"Stores         : {', '.join(df['store_name'].dropna().unique())}\n\n"

    # add column data
    out += "Columns:\n"

    for col_name in df.columns:
      out += f"\t{col_name}\n"

    out += (
    "\n----------------------------------\n"
    "Rows per COICOP Level 1 Categories\n"
    "----------------------------------\n"
    )

    # add coicop data
    coicop_counts = df["coicop_level_1"].value_counts()
    coicop_counts = coicop_counts.sort_index()

    for coicop_level, counts in coicop_counts.items():
      out += f"\t{coicop_level}: {counts:<10} {(counts / df.shape[0]):.4f}\n"

    out += (
    "\n----------------------------------\n"
    "Rows per Store\n"
    "----------------------------------\n"
    )

    # add coicop data
    store_counts = df["store_name"].value_counts()
    store_counts = store_counts.sort_index()

    for store_name, counts in store_counts.items():
      out += f"\t{store_name:<15}: {counts:<10} {(counts / df.shape[0]):.4f}\n"

    with open(metadata_output_path, 'w') as fp:
      fp.write(out)
      
if __name__ == "__main__":
  df_stores = [] # all stores

  print("Loading datasets...")
  for store_name in config.STORES:
    dataset_fn = f"ssi_{store_name}_revenue.parquet"
    dataset_path = os.path.join(config.SOURCE_DATA_DIR, dataset_fn)

    df = pd.read_parquet(dataset_path, columns=LOAD_COLUMNS)
    df_stores.append(df)

  df_stores = pd.concat(df_stores)
  df_stores = df_stores.rename(COL_NAME_MAPPING, axis=1)

  df_stores_train, df_stores_tests = split_train_test(df_stores) # df_stores_tests contains test sets for mulitple periods

  print("Saving Training data set...")
  out_base_fn = f"{'_'.join(config.STORES)}_incl_999999.parquet"

  df_stores_train  = preprocess(df_stores_train, filter_records=True, drop_999999=False)
  out_train_fn = f"train_{out_base_fn}"
  write_dataset(df_stores_train, out_fn=out_train_fn)

  print("Saving Test data sets...")
  for df_period, df_test in df_stores_tests.items():
    df_test = preprocess(df_test, filter_records=True, drop_999999=False)
    out_test_fn = f"test_{df_period}_{out_base_fn}"
    write_dataset(df_test, out_fn=out_test_fn)

  print("Saving Full data sets...")
  df_stores_full  = preprocess(df_stores, filter_records=True, drop_999999=False)
  out_fn_full = f"full_{out_base_fn}"
  write_dataset(df_stores_full, out_fn=out_fn_full)

#  df_stores_full_dup  = preprocess(df_stores, group_by_period=False)
#  out_fn_full_dup = f"full_dup_{out_base_fn}.parquet"
#  write_dataset(df_stores_full_dup, out_fn=out_fn_full_dup)


  


