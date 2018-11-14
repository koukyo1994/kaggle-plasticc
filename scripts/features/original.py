import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.linear_model import LinearRegression


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
    template = pd.DataFrame({
        "object_id": new_df.object_id.unique(),
        "passband0": 0,
        "passband1": 0,
        "passband2": 0,
        "passband3": 0,
        "passband4": 0,
        "passband5": 0
    })
    for objid in tqdm(new_df.object_id.unique()):
        obj_df = new_df.query("object_id == @objid")[[
            "mjd", "cluster", "passband", "flux_normalized"
        ]]
        passbands = [[] for _ in range(6)]
        for cl in obj_df.cluster.unique():
            cluster_df = obj_df.query("cluster == @cl")
            for ps in cluster_df.passband.unique():
                ps_df = cluster_df.query("passband == @ps")
                if ps_df.shape[0] <= 1:
                    passbands[ps].append(0)
                    continue
                lr.fit(ps_df["mjd"].values.reshape([-1, 1]),
                       ps_df["flux_normalized"].values.reshape([-1, 1]))
                passbands[ps].append(np.abs(lr.coef_)[0][0])
        passbands = [np.mean(p) for p in passbands]
        for i, ps in enumerate(passbands):
            template.loc[template.query("object_id == @objid").
                         index, f"passband{i}"] = ps
    return template


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
