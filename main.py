import argparse
from coding_standards.models.xgb import func
from coding_standards.data_transformations import data_display, pipeline, FeatureEncoder, BalanceFeature
from coding_standards import get_data
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", default=False, action="store_true")
    parser.add_argument("-a", default=2, type=int)
    args = parser.parse_args()

    pd.options.display.max_columns = None
    data = get_data()
    data = pipeline.fit_transform(data)
    # encoder = FeatureEncoder(["gender", "country"])
    # encoder.fit(data=data)
    # data = encoder.transform(data)
    #
    # encoder = BalanceFeature()
    # encoder.fit(data)
    # data = encoder.transform(data)

    data = pipeline.inverse_transform(data)
    print(data)