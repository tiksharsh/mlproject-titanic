import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from sklearn import preprocessing

# from sklearn.linear_model import SGDClassifier


def feature_engineering(raw_df):
    df = raw_df.copy()
    df["Cabin"] = df["Cabin"].apply(lambda x: x[:1] if x is not np.nan else np.nan)
    df["Family"] = df["SibSp"] + df["Parch"]
    logging.info(f"Inside the feature engineering and dataframe")
    logging.info("-" * 80)
    return df

def to_category(train_df, test_df):
    cat = ["Sex", "Cabin", "Embarked"]
    for col in cat:
        le = preprocessing.LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        logging.info(f"Inside to_category")
        logging.info("-" * 80)
        return train_df, test_df
    
