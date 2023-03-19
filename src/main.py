import os 
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

import joblib
import dagshub

sys.path.append("src")
from src.logger import logging
from src.exception import CustomException

from sklearn import preprocessing

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


@dataclass

class DataIngestionConfig:
    drop_cols = ["Name", "SibSp", "Parch", "Ticket"]
    obj_col = "Survived"
    train_df_path: str = os.path.join('Data',"train.csv")
    test_df_path: str = os.path.join('Data', "test.csv")
    sub_df_path: str = os.path.join('Data',"sample_submission.csv")
    logging.info(f"Inside DataIngestionConfig {train_df_path}")
    logging.info('-'*80)
    # train_df_path = "/Users/Wolverine/Desktop/Machine_Learning/DAGsHub/DagsHub-TitanicDataSet/Data/train.csv"
    # test_df_path = "/Users/Wolverine/Desktop/Machine_Learning/DAGsHub/DagsHub-TitanicDataSet/Data/test.csv"
    # sub_df_path = "/Users/Wolverine/Desktop/Machine_Learning/DAGsHub/DagsHub-TitanicDataSet/Data/sample_submission.csv"

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def feature_engineering(self,raw_df):
        df = raw_df.copy()
        df["Cabin"] = df["Cabin"].apply(lambda x: x[:1] if x is not np.nan else np.nan)   
        df["Family"] = df["SibSp"] + df["Parch"]
        # logger.info(f"Inside the feature engineering and dataframe: {df}")
        # logging.info(f"Inside feature_engineering")
        # logging.info('-'*80)
        return df

    def fit_model(self,train_X, train_y, random_state=42):
        clf = SGDClassifier(loss="modified_huber", random_state=random_state)
        clf.fit(train_X, train_y)
        # logging.info(f"Inside fit_model")
        # logging.info('-'*80)
        return clf

    def to_category(self,train_df, test_df):
        cat = ["Sex", "Cabin", "Embarked"]
        for col in cat:
            le = preprocessing.LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col])
            test_df[col] = le.transform(test_df[col])
            # logging.info(f"Inside to_category")
            # logging.info('-'*80)
        return train_df, test_df

    def eval_model(self,clf, X, y):
        y_proba = clf.predict_proba(X)[:, 1]
        y_pred = clf.predict(X)
        # logging.info(f"Inside eval_model")
        # logging.info('-'*80)
        return {
            "roc_auc": roc_auc_score(y, y_proba),
            "average_precision": average_precision_score(y, y_proba),
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
        }

    def submission(self,clf, X):
        sub = pd.read_csv(self.ingestion_config.sub_df_path)
        sub[self.ingestion_config.obj_col] = clf.predict(X)
        # logging.info(f"Inside submission")
        # logging.info('-'*80)
        sub.to_csv("Submission/submission.csv", index=False)

        
    def model_run(self):
        logging.info("Loading data...")
        df_train = pd.read_csv(self.ingestion_config.train_df_path, index_col="PassengerId")
        df_test = pd.read_csv(self.ingestion_config.test_df_path, index_col="PassengerId")
        
        logging.info("Engineering features...")
        y = df_train[self.ingestion_config.obj_col]
        X = self.feature_engineering(df_train).drop(self.ingestion_config.drop_cols + [self.ingestion_config.obj_col], axis=1)
        
        test_df = self.feature_engineering(df_test).drop(self.ingestion_config.drop_cols, axis=1)
        X, test_df = self.to_category(X, test_df)
        X.fillna(0, inplace=True)
        test_df.fillna(0, inplace=True)
        
        with dagshub.dagshub_logger() as logger:
            
            try:
                logging.info("Training model...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                model = self.fit_model(X_train, y_train)
                
                logging.info("Saving trained model...")
                joblib.dump(model, "Model/model.joblib")
                logger.log_hyperparams(model_class=type(model).__name__)
                logger.log_hyperparams({"model": model.get_params()})
                
                logging.info("Evaluating model...")
                train_metrics = self.eval_model(model, X_train, y_train)
                
                
                logging.info(f"Train metrics: {train_metrics}")
                logger.log_metrics({f"train__{k}": v for k, v in train_metrics.items()})
                test_metrics = self.eval_model(model, X_test, y_test)
                
                logging.info(f"Test metrics: {test_metrics}")
                logger.log_metrics({f"test__{k}": v for k, v in test_metrics.items()})
                
                logging.info("Creating Submission File...")
                self.submission(model, test_df)
            except Exception as e:
                logging.info(f"Indside model_run exception!! {e}")
                CustomException(e,sys)

        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.model_run()
    