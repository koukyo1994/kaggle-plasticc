import pandas as pd

from features.from_kernel import basic, with_cluster
from features.original import cluster_mean_diff, passband_std_difference
from features.original import linear_slope, num_outliers


def get_full(df, meta):
    agg_basic = basic(df)
    agg_cluster = with_cluster(df)

    cl_mean_diff = cluster_mean_diff(df)
    ps_std_diff = passband_std_difference(df)
    lin_sl = linear_slope(df)
    num_out = num_outliers(df)

    full = pd.merge(agg_basic, agg_cluster, how="left", on="object_id")
    full = pd.merge(full, cl_mean_diff, how="left", on="object_id")
    full = pd.merge(full, ps_std_diff, how="left", on="object_id")
    full = pd.merge(full, lin_sl, how="left", on="object_id")
    full = pd.merge(full, num_out, how="left", on="object_id")

    full = pd.merge(full, meta, how="left", on="object_id")
    if "target" in full.columns:
        full.drop("target", axis=1, inplace=True)
    return full


def train_data(df, meta):
    full = get_full(df, meta)
    y = meta.target
    classes = sorted(y.unique())
    class_weight = {c: 1 for c in classes}

    for c in [64, 15]:
        class_weight[c] = 2
    oof_df = full[["object_id"]]
    del full['object_id'], full['distmod'], full['hostgal_specz']
    del full['ra'], full['decl'], full['gal_l'], full['gal_b'], full['ddf']
    return full, y, classes, class_weight, oof_df
