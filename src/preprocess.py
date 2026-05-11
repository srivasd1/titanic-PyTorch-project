# src/preprocess.py

import pandas as pd

class Preprocessor:

    def __init__(self, config):
        self.config = config

    def transform(self, df):

        df = df.copy()

        # drop columns
        df.drop(self.config["drop_columns"], axis=1, errors="ignore", inplace=True)

        # missing values
        for col, method in self.config["fillna"].items():

            if method == "median":
                df[col] = df[col].fillna(df[col].median())

            elif method == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])


        # encoding
        print(self.config["sex_mapping"])
        df["Sex"] = df["Sex"].map({"male": 0, "female":1})

        df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

        # align columns
        for col in self.config["feature_columns"]:
            if col not in df:
                df[col] = 0

        df = df[self.config["feature_columns"]]
        df = df.astype("float32")
	
        return df
