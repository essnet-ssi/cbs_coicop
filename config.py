import os

STORES = ["lidl", "ah", "jumbo", "plus"]

SOURCE_DATA_DIR = os.path.join('/', "data", "projecten", "ssi", "preprocessing", "05-final")
OUTPUT_DATA_DIR = os.path.join('..', "data")

OUTPUT_GRAPHICS_DIR = os.path.join('..', "graphics")

RESULTS_DIR  = os.path.join('..', "results")
PIPELINE_DIR = os.path.join("..", "pipelines")

FEATURE_EXTRACTION_MODEL_DIR = os.path.join('.', "feature_extraction_models")
FEATURE_EXTRACTION_LABSE_PATH = os.path.join(FEATURE_EXTRACTION_MODEL_DIR, "hf_labse_model")

SEED = 42

