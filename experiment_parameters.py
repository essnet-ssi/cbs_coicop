import config

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import OneHotEncoder

from coicop_experiment import CoicopExperiment
from feature_extraction import LabseExtractor, SpacyExtractor
from string_matching import ExactStringMatcher
from models import FastTextClassifier
from sklearn.compose import ColumnTransformer

PREDICT_LEVEL: int = 5
EXPERIMENT_TYPE: str = "store" # time | "store"

SPLIT_TESTS_BY: str | None = None # ONLY RELEVANT FOR TIME EXPERIMENTS. col name to split test sets by.
RESULTS_FN: str = "output.file"
INCLUDE_999999 = True

#WEIGHTS_COL_NAME: str = "revenue"
WEIGHTS_COL_NAME: str = None

SAVE_PIPELINE: bool = True

EXPERIMENT_ANNOTATIONS: str = ""
EXPERIMENT_ANNOTATIONS += "with 999999. " if INCLUDE_999999 else  "without 999999."
if WEIGHTS_COL_NAME:
  EXPERIMENT_ANNOTATIONS += f"weighted by {WEIGHTS_COL_NAME}."

# sampling strategies: random_sample | stratify_store | None

EXPERIMENTS: list[CoicopExperiment] = [
  CoicopExperiment(
    pipeline=Pipeline([
      ("tfidf", TfidfVectorizer(input="content", binary=True, lowercase=True, ngram_range=(2, 4), analyzer="char")),
      ("clf", DummyClassifier(strategy="prior", random_state=config.SEED))
    ]),
    param_grid={
    },
    train_sample_strategy=None
  ),

  CoicopExperiment(
    pipeline=Pipeline([
      ("tfidf", TfidfVectorizer(input="content", binary=True, lowercase=True, ngram_range=(2, 4), analyzer="char")),
      ("clf", LogisticRegression(solver="saga", random_state=config.SEED, n_jobs=-1))
    ]),
    param_grid={
      "clf__C": [1000, 100, 10, 1, .1, .001, .0001],
    },
    train_sample_strategy="random_sample"
  ),
  CoicopExperiment(
    pipeline=Pipeline([
      ("tfidf", TfidfVectorizer(input="content", binary=True, lowercase=True, ngram_range=(2, 4), analyzer="char")),
      ("clf", RandomForestClassifier(n_jobs=-1, random_state=config.SEED))
    ]),
    param_grid={
      "clf__max_features": ["log2", "sqrt"]
    },
    train_sample_strategy=None # sampling is done by the classifier itself
  ),
  CoicopExperiment(
    pipeline=Pipeline([
      ("tfidf", TfidfVectorizer(input="content", binary=True, lowercase=True, ngram_range=(2, 4), analyzer="char")),
      ("clf", MLPClassifier(random_state=config.SEED, solver="adam", max_iter=50))
    ]),
    param_grid={
      "clf__alpha": [.01, .001, .0001, .00001, .000001]
    },
    train_sample_strategy="random_sample"
  ),
  CoicopExperiment(
    pipeline=Pipeline([
      ("tfidf", TfidfVectorizer(input="content", binary=True, lowercase=True, ngram_range=(2, 4), analyzer="char")),
      ("clf", MultinomialNB())
    ]),
    param_grid={
      "clf__alpha": [1., .1, .01, .001, .0001, .00001]
    },
    #train_sample_strategy="random_sample"
  ),
  CoicopExperiment(
    pipeline=Pipeline([
      @ Doesnt work properly with gridsearchcv yet!
      ("clf", ExactStringMatcher(not_found_value='NOT_FOUND'))
    ]),
    param_grid={
    },
    train_sample_strategy=None # sampling is done by the classifier itself
  ),
  CoicopExperiment(
    pipeline=Pipeline([
      ("tfidf", TfidfVectorizer(input="content", binary=True, lowercase=True, ngram_range=(2, 4), analyzer="char")),
      ("clf", SGDClassifier(n_jobs=-1, random_state=config.SEED))
    ]),
    param_grid={
      "clf__loss": ["log_loss"],
      "clf__alpha": [1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    },
    train_sample_strategy=None
  ),
  CoicopExperiment(
    pipeline=Pipeline([
      ("tfidf", TfidfVectorizer(input="content", binary=True, lowercase=True, ngram_range=(2, 4), analyzer="char")),
      ("clf", DummyClassifier(strategy="prior", random_state=config.SEED))
    ]),
    param_grid={
    },
    train_sample_strategy=None
  ),
  CoicopExperiment(
    pipeline=Pipeline([
      ("clf", FastTextClassifier())
    ]),
    param_grid={
      "clf__lr": [0.1],
      "clf__epoch": [30],
      "clf__bucket": [4_000_000],
      "clf__loss": ["softmax"],
      "clf__thread": [20],
      "clf__preprocess": [True, False],
      "clf__minn": [2],
      "clf__maxn": [4]
    },
    train_sample_strategy=None
  ),
]
 
