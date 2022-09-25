import argparse
from coding_standards.models.xgb import func
from coding_standards.data_transformations import data_display, pipeline, pipeline_numerics
from coding_standards import get_data
from coding_standards import EDA
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", default=False, action="store_true")
    parser.add_argument("-a", default=2, type=int)
    args = parser.parse_args()

    pd.options.display.max_columns = None
    data = get_data()

    train, test = train_test_split(data,
                                   test_size=0.4,
                                   stratify=data.churn,
                                   random_state=42)

    validation, test = train_test_split(test,
                                   test_size=0.5,
                                   stratify=test.churn,
                                   random_state=42)

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train_t = pipeline.fit_transform(train.reset_index(drop=True))
    validation_t = pipeline.transform(validation)
    test_t = pipeline.transform(test)

    # we avoid the one-hot encoding since it would skew results for some EDA such as VIF
    train_numerics = pipeline_numerics.fit_transform(train.reset_index(drop=True))
    EDA(train_numerics)

