from pathlib import Path
import pandas as pd
import os
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# import and display
def data_display():
    path = Path(__file__).parent.as_posix()
    # return pd.read_csv(os.path.join(path, "data", "bank_customer_churn_prediction.csv"))
    return pd.read_csv(os.path.join(path, "data", "bank_customer_churn_prediction.csv"))



def get_data():
    current_path = Path(__file__).parent.as_posix()
    data_path = os.path.join(current_path, "data")
    data_files = [x for x in os.listdir(data_path) if ".csv" in x]

    if len(data_files) == 1:
        return pd.read_csv(os.path.join(data_path, data_files[0]))
    elif len(data_files) > 1:
        logging.exception(f"Too many data files provided. Please remove all but one from {data_path}")
    else:
        logging.exception(f"No data provided. Please add it under {data_path}")



class FeatureEncoder():
    def __init__(self, encoded_columns: list[str]) -> None:
        self.encoded_columns = encoded_columns
        self.encoders = {column: LabelEncoder() for column in encoded_columns}

    def fit(self, data, y=None) -> None:
        for column in self.encoded_columns:
            self.encoders[column].fit(data[column])
        return self

    def transform(self, data, y=None):
        for column in self.encoded_columns:
            encoded_col = self.encoders[column].transform(data[column])
            data[column] = encoded_col
        return data

    def inverse_transform(self, data):
        for column in self.encoded_columns:
            decoded_col = self.encoders[column].inverse_transform(data[column])
            data[column] = decoded_col
        return data

    def classes_(self, column_name):
        return self.encoders[column_name].classes_



class OneHotMultiple():
    def __init__(self,
                 encoded_columns: list[str],
                 name_divider: str = "==") -> None:
        self.encoded_columns = encoded_columns
        self.encoders = {column: OneHotEncoder(handle_unknown="ignore",
                                               sparse=False) for column in encoded_columns}
        self.name_divider = name_divider

    def fit(self, data, y=None) -> None:
        for column in self.encoded_columns:
            self.encoders[column].fit(data[[column]])
        return self

    def transform(self, data, y=None) -> pd.DataFrame:
        for column in self.encoded_columns:
            encoded_col = self.encoders[column].transform(data[[column]])
            data.drop(columns=column, inplace=True)
            categories = self.encoders[column].categories_[0]
            extended_col = pd.DataFrame(encoded_col, columns=[f"{column}{self.name_divider}{x}" for x in categories])
            data = pd.concat((data, extended_col), axis=1)
        return data

    def inverse_transform(self, data) -> pd.DataFrame:
        for column in self.encoded_columns:
            relevant_cols = [f"{column}{self.name_divider}{cat}" for cat in \
                                                self.encoders[column].categories_[0]]
            data_subset = data[relevant_cols]
            inverse_transformed = self.encoders[column].inverse_transform(data_subset)
            inverse_transformed = pd.DataFrame(inverse_transformed, columns=[column])
            data.drop(columns=relevant_cols, inplace=True)
            data = pd.concat((data, inverse_transformed), axis=1)
        return data

    def categories_(self, column_name):
        return self.encoders[column_name].categories_



class BalanceFeature():
    def __init__(self):
        pass

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        data["balance/salary"] = data.balance / data.estimated_salary
        return data

    def inverse_transform(self, data):
        data = data.drop(columns=["balance/salary"])
        return data



# add features based on customer history
class CustomerFeatures():
    def __init__(self):
        pass


# as placeholder simply remove the customer id feature
class CustomerFeaturesPlaceholder():
    def __init__(self, to_drop):
        self.to_drop = to_drop
        self.recorded_col = None

    def fit(self, data, y=None):
        self.recorded_col = data[self.to_drop].to_list()
        return self

    def transform(self, data, y=None):
        return data.drop(columns=self.to_drop)

    def inverse_transform(self, data):
        data[self.to_drop] = self.recorded_col
        return data



# pipeline = Pipeline([("categorical_encoder", FeatureEncoder(["country", "gender"])),
#                      ("balance_feature", BalanceFeature())])

pipeline = Pipeline([("categorical_encoder", FeatureEncoder(["gender"])),
                     ("onehot_encoder", OneHotMultiple(["country"])),
                     ("customer_features", CustomerFeaturesPlaceholder("customer_id")),
                     ("balance_feature", BalanceFeature())])

pipeline_numerics = Pipeline([("categorical_encoder", FeatureEncoder(["gender", "country"])),
                              ("customer_features", CustomerFeaturesPlaceholder("customer_id")),
                              ("balance_feature", BalanceFeature())])