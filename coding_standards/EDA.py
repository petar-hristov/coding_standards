import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas_profiling
from sklearn.ensemble import RandomForestClassifier


def plot_correlations(df, save_path="results"):
    sns.set(rc={"figure.figsize": (15, 10)})
    sns.heatmap(df.corr(), annot=True)
    Path(os.path.join(save_path, "plots")).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_path, "plots", "train_correlations_plot.png"))

def get_dtypes(df):
    print("The dtype of each column:")
    print(df.dtypes, "\n")


def get_missing_vals(df):
    print("The proportions of missing values for each column:")
    print(df.isnull().sum() / len(df), "\n")


def get_nuniques(df):
    print("The number of unique values in each column:")
    print(df.nunique(), "\n")


def get_VIFs(df, label_col="churn"):
    df_nolabel = df.drop(columns=label_col)

    vifs = dict()
    for ind, col in enumerate(df_nolabel.columns):
        vif = variance_inflation_factor(df_nolabel, exog_idx=ind)
        vifs[col] = [vif]

    print("The variance inflation factors of each feature:")
    print(pd.DataFrame(vifs), "\n")


def get_profiling_report(df, save_path="results"):
    profile = pandas_profiling.ProfileReport(df, title="Pandas Profiling Report")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    profile.to_file(os.path.join(save_path, "train_set_EDA_report.html"))


def get_feature_importances(df):
    model = RandomForestClassifier()
    X = df.drop(columns="churn")
    y = df["churn"]
    model.fit(X=X,
              y=y)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({col: [importance] for col, importance in zip(X.columns, importances)})
    importance_df = importance_df.T.rename(columns={0: "importance"})

    print("Feature importances:")
    print(importance_df.sort_values("importance", ascending=False), "\n")

def EDA(df):

    plot_correlations(df)
    get_dtypes(df)
    get_missing_vals(df)
    get_nuniques(df)
    get_VIFs(df, label_col="churn")
    get_profiling_report(df)
    get_feature_importances(df)