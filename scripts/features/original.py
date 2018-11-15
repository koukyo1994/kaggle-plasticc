import pandas as pd

from tqdm import tqdm
from sklearn.linear_model import LinearRegression

tqdm.pandas(desc="apply progress")


def cluster_mean_diff(df):
    new_df = df.groupby(["object_id", "cluster"]).agg({
        "flux": ["mean", "max", "min"]
    })
    new_df.columns = pd.Index(
        [e[0] + "_" + e[1] for e in new_df.columns.tolist()])
    new_df["normalized_mean"] = new_df["flux_mean"] / (
        new_df["flux_max"] - new_df["flux_min"])
    new_df.reset_index(inplace=True)
    return new_df.groupby("object_id").agg({"normalized_mean": "std"})


def passband_std_difference(df):
    std_df = df.groupby(["object_id", "cluster", "passband"]).agg({
        "flux": "std"
    }).reset_index().groupby(["object_id",
                              "passband"])["flux"].mean().reset_index()
    std_df_max = std_df.groupby("object_id")["flux"].max()
    std_df_min = std_df.groupby("object_id")["flux"].min()
    return (std_df_max / std_df_min).reset_index()


def linear_slope(df):
    new_df = df.groupby(["object_id", "cluster", "passband"]).agg({
        "flux": ["max", "min"]
    })
    new_df.columns = pd.Index([e[0] + "_" + e[1] for e in new_df.columns])
    new_df.reset_index(inplace=True)
    new_df["flux_range"] = new_df["flux_max"] - new_df["flux_min"]
    new_df = pd.merge(
        df, new_df, how="left", on=["object_id", "cluster", "passband"])
    new_df["flux_normalized"] = new_df["flux"] / new_df["flux_range"]
    lr = LinearRegression()

    def get_coef_(x):
        if x.shape[0] <= 1:
            return 0
        lr.fit(
            x.mjd.values.reshape([-1, 1]),
            x.flux_normalized.values.reshape([-1, 1]))
        return lr.coef_[0][0]

    new_df = new_df.groupby(
        ["object_id", "cluster", "passband"],
        as_index=False)["object_id", "cluster", "passband", "mjd",
                        "flux_normalized"].progress_apply(
                            lambda x: get_coef_(x)).unstack().groupby(
                                "object_id", as_index=False).mean()
    new_df.columns = pd.Index(
        [new_df.columns.name + e for e in new_df.columns])
    return new_df


def num_outliers(df):
    new_df = df.groupby("object_id").agg({"flux": ["mean", "std"]})
    new_df.columns = pd.Index([e[0] + "_" + e[1] for e in new_df.columns])
    new_df["upper_sigma"] = new_df["flux_mean"] + new_df["flux_std"]
    new_df["upper_2sigma"] = new_df["flux_mean"] + 2 * new_df["flux_std"]
    new_df["lower_sigma"] = new_df["flux_mean"] - new_df["flux_std"]
    new_df["lower_2sigma"] = new_df["flux_mean"] - 2 * new_df["flux_std"]
    new_df.drop(["flux_mean", "flux_std"], axis=1, inplace=True)
    new_df = pd.merge(df, new_df, how="left", on="object_id")
    new_df["outside_sigma"] = (
        (new_df["flux"] > new_df["upper_sigma"]) |
        (new_df["flux"] < new_df["lower_sigma"])).astype(int)
    new_df["outside_2sigma"] = (
        (new_df["flux"] > new_df["upper_2sigma"]) |
        (new_df["flux"] < new_df["lower_2sigma"])).astype(int)

    return_df = new_df.groupby("object_id").agg({
        "outside_sigma": "sum",
        "outside_2sigma": "sum"
    })
    return_df.reset_index(inplace=True)
    return return_df
