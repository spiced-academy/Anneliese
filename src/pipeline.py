import pandas as pd
from sklearn.pipeline import Pipeline, FunctionTransformer
from src.data_cleaning import clean_data
from src.feature_engineering import add_features

def build_pipeline() -> Pipeline:
    """
    Build a scikit-learn pipeline for cleaning and feature engineering.
    Returns:
        sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline([
        ('cleaning', FunctionTransformer(clean_data, validate=False)),
        ('feature_engineering', FunctionTransformer(add_features, validate=False))
    ])
    return pipeline
