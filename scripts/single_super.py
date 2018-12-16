import os
import gc
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from datetime import datetime as dt
from functools import partial
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from tsfresh.feature_extraction import extract_features
from numba import jit

tqdm.pandas(desc="apply progress")
gc.enable()

ver = '4-0-2'


@jit
def haversine_plus(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) from
    #https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    # Convert decimal degrees to Radians:
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    # Implementing Haversine Formula:
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(
        np.power(np.sin(np.divide(dlat, 2)), 2),
        np.multiply(
            np.cos(lat1),
            np.multiply(np.cos(lat2), np.power(np.sin(np.divide(dlon, 2)),
                                               2))))

    haversine = np.multiply(2, np.arcsin(np.sqrt(a)))
    return {
        'haversine': haversine,
        'latlon1': np.subtract(
            np.multiply(lon1, lat1), np.multiply(lon2, lat2)),
    }


@jit
def process_flux(df):
    flux_ratio_sq = np.power(df['flux'].values / df['flux_err'].values, 2.0)

    df_flux = pd.DataFrame(
        {
            'flux_ratio_sq': flux_ratio_sq,
            'flux_by_flux_ratio_sq': df['flux'].values * flux_ratio_sq,
        },
        index=df.index)

    return pd.concat([df, df_flux], axis=1)


@jit
def process_flux_agg(df):
    flux_w_mean = df['flux_by_flux_ratio_sq_sum'].values / df[
        'flux_ratio_sq_sum'].values
    flux_diff = df['flux_max'].values - df['flux_min'].values

    df_flux_agg = pd.DataFrame(
        {
            'flux_w_mean': flux_w_mean,
            'flux_diff1': flux_diff,
            'flux_diff2': flux_diff / df['flux_mean'].values,
            'flux_diff3': flux_diff / flux_w_mean,
        },
        index=df.index)

    return pd.concat([df, df_flux_agg], axis=1)


def make_diff_feature(df):
    tmp = df.groupby(['object_id', 'passband',
                      'mjd'])['flux'].sum().reset_index()

    tmp['flux_diff'] = tmp['flux'] - tmp['flux'].shift(1)
    multi_id_list = (tmp['object_id'].astype(str) + '-' +
                     tmp['passband'].astype(str)).values

    drop_index = []
    prev_val = 'hoge'
    for index, val in enumerate(multi_id_list):
        if val != prev_val:
            drop_index.append(index)
        prev_val = val

    use_index = list(set(tmp.index) - set(drop_index))
    tmp = tmp.iloc[use_index, :]
    diff_df = tmp.drop('flux', axis=1)

    return diff_df


def featurize(df, df_meta, aggs, fcp, n_jobs=36):
    """
    Extracting Features from train set
    Features from olivier's kernel
    very smart and powerful feature that is generously given here https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    per passband features with tsfresh library. fft features added to capture periodicity https://www.kaggle.com/c/PLAsTiCC-2018/discussion/70346#415506
    """

    df = process_flux(df)

    agg_df = df.groupby('object_id').agg(aggs)
    agg_df.columns = [
        '{}_{}'.format(k, agg) for k in aggs.keys() for agg in aggs[k]
    ]
    agg_df = process_flux_agg(agg_df)  # new feature to play with tsfresh
    per_passband_aggs = {
        "flux": ["min", "max", "mean", "std"],
        "flux_ratio_sq": ["sum", "skew"],
        "flux_by_flux_ratio_sq": ["sum", "skew"]
    }
    per_pass_agg_df = df.groupby(["object_id",
                                  "passband"]).agg(per_passband_aggs)
    per_pass_agg_df.columns = pd.Index(
        [e[0] + "_" + e[1] for e in per_pass_agg_df.columns])
    per_pass_agg_df["flux_diff"] = per_pass_agg_df[
        "flux_max"] - per_pass_agg_df["flux_min"]
    per_pass_agg_df["flux_diff2"] = (
        per_pass_agg_df["flux_max"] -
        per_pass_agg_df["flux_min"]) / per_pass_agg_df["flux_mean"]
    per_pass_agg_df["flux_w_mean"] = per_pass_agg_df[
        "flux_by_flux_ratio_sq_sum"] / per_pass_agg_df["flux_ratio_sq_sum"]
    per_pass_agg_df["flux_dif3"] = (
        per_pass_agg_df["flux_max"] -
        per_pass_agg_df["flux_min"]) / per_pass_agg_df["flux_w_mean"]
    per_pass_agg_df = per_pass_agg_df.unstack()
    per_pass_agg_df.columns = pd.Index(
        [str(e[1]) + "__" + e[0] for e in per_pass_agg_df.columns])

    basic_columns = [
        f"{i}__{j}" for i in range(6) for j in [
            "flux_min", "flux_max", "flux_mean", "flux_std",
            "flux_ratio_sq_sum", "flux_ratio_sq_skew", "flux_w_mean",
            "flux_diff2"
        ]
    ]
    per_pass_agg_df.drop(basic_columns, axis=1, inplace=True)

    agg_df = pd.merge(agg_df, per_pass_agg_df, how="left", on="object_id")

    agg_flux_diff = agg_df.reset_index()[["object_id", "flux_diff1"]]
    df2 = pd.merge(df, agg_flux_diff, how="left", on="object_id")
    df2["flux_norm"] = df2.flux / df2.flux_diff1
    del df2["flux"]

    # Add more features with
    agg_df_ts_flux_passband = extract_features(
        df,
        column_id='object_id',
        column_sort='mjd',
        column_kind='passband',
        column_value='flux',
        default_fc_parameters=fcp['flux_passband'],
        n_jobs=n_jobs)

    agg_df_ts_flux = extract_features(
        df,
        column_id='object_id',
        column_value='flux',
        default_fc_parameters=fcp['flux'],
        n_jobs=n_jobs)

    agg_df_ts2 = extract_features(
        df2,
        column_id="object_id",
        column_sort="mjd",
        column_kind="passband",
        column_value="flux_norm",
        default_fc_parameters=fcp["fcp2"],
        n_jobs=n_jobs)

    agg_df_ts_flux_by_flux_ratio_sq = extract_features(
        df,
        column_id='object_id',
        column_value='flux_by_flux_ratio_sq',
        default_fc_parameters=fcp['flux_by_flux_ratio_sq'],
        n_jobs=n_jobs)

    # Add smart feature that is suggested here https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    # dt[detected==1, mjd_diff:=max(mjd)-min(mjd), by=object_id]
    df_det = df[df['detected'] == 1].copy()
    agg_df_mjd = extract_features(
        df_det,
        column_id='object_id',
        column_value='mjd',
        default_fc_parameters=fcp['mjd'],
        n_jobs=n_jobs)
    agg_df_mjd['mjd_diff_det'] = agg_df_mjd[
        'mjd__maximum'].values - agg_df_mjd['mjd__minimum'].values
    del agg_df_mjd['mjd__maximum'], agg_df_mjd['mjd__minimum']
    agg_df_ts2.columns = pd.Index([e + "_norm" for e in agg_df_ts2.columns])
    agg_df_ts_flux_passband.index.rename('object_id', inplace=True)
    agg_df.index.rename('object_id', inplace=True)
    agg_df_ts_flux.index.rename('object_id', inplace=True)
    agg_df_ts_flux_by_flux_ratio_sq.index.rename('object_id', inplace=True)
    agg_df_mjd.index.rename('object_id', inplace=True)
    agg_df_ts = pd.concat([
        agg_df, agg_df_ts2, agg_df_ts_flux_passband, agg_df_ts_flux,
        agg_df_ts_flux_by_flux_ratio_sq, agg_df_mjd
    ],
                          axis=1).reset_index()

    result = agg_df_ts.merge(right=df_meta, how='left', on='object_id')
    return result


def diff_featurize(diff_df, df_meta, diff_aggs, fcp, n_jobs=36):
    """
    Extracting Features from train set
    Features from olivier's kernel
    very smart and powerful feature that is generously given here https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    per passband features with tsfresh library. fft features added to capture periodicity https://www.kaggle.com/c/PLAsTiCC-2018/discussion/70346#415506
    """

    # df = train.copy()

    # diff_df = process_flux(diff_df)

    diff_agg_df = diff_df.groupby('object_id').agg(diff_aggs)
    diff_agg_df.columns = [
        '{}_{}'.format(k, agg) for k in diff_aggs.keys()
        for agg in diff_aggs[k]
    ]
    # diff_agg_df = process_flux_agg(diff_agg_df) # new feature to play with tsfresh

    # Add more features with
    diff_agg_df_ts_flux_passband = extract_features(
        diff_df,
        column_id='object_id',
        column_sort='mjd',
        column_kind='passband',
        column_value='flux_diff',
        default_fc_parameters=fcp['flux_passband'],
        n_jobs=n_jobs)

    diff_agg_df_ts_flux = extract_features(
        diff_df,
        column_id='object_id',
        column_value='flux_diff',
        default_fc_parameters=fcp['flux'],
        n_jobs=n_jobs)

    diff_agg_df_ts_flux_passband.index.rename('object_id', inplace=True)
    diff_agg_df_ts_flux_passband.columns = [
        column + '_diff' for column in diff_agg_df_ts_flux_passband.columns
    ]
    diff_agg_df_ts_flux.index.rename('object_id', inplace=True)
    # agg_df_ts_flux_by_flux_ratio_sq.index.rename('object_id', inplace=True)
    # agg_df_mjd.index.rename('object_id', inplace=True)
    diff_agg_df_ts = pd.concat(
        [
            diff_agg_df,
            diff_agg_df_ts_flux_passband,
            diff_agg_df_ts_flux,
            # agg_df_ts_flux_by_flux_ratio_sq,
            # agg_df_mjd
        ],
        axis=1).reset_index()

    # result = agg_df_ts.merge(right=df_meta, how='left', on='object_id')
    result = diff_agg_df_ts
    return result


def process_meta(filename):
    meta_df = pd.read_csv(filename)

    meta_dict = dict()

    # id trick
    for i in [22, 27]:
        meta_dict['object_id_div_{}'.format(i)] = np.mod(
            meta_df['object_id'].values, i)

    # distance
    meta_dict.update(
        haversine_plus(meta_df['ra'].values, meta_df['decl'].values,
                       meta_df['gal_l'].values, meta_df['gal_b'].values))
    #
    meta_dict['hostgal_photoz_certain'] = np.multiply(
        meta_df['hostgal_photoz'].values,
        np.exp(meta_df['hostgal_photoz_err'].values))

    meta_df = pd.concat(
        [meta_df, pd.DataFrame(meta_dict, index=meta_df.index)], axis=1)
    return meta_df


def add_rank_bottom_and_top(df, feature_name):
    objid = ["object_id"]
    columns = [f"{i}{feature_name}" for i in range(6)]
    partial_df = df[objid + columns]
    partial_values = partial_df.melt(
        id_vars=objid, value_vars=columns).sort_values(["object_id", "value"])

    top_and_bottom = partial_values.groupby("object_id").agg({
        "variable": ["first", "last"]
    })
    top_and_bottom.columns = ["top" + feature_name, "bottom" + feature_name]
    for i, n in zip(["0", "1", "2", "3", "4", "5"], columns):
        top_and_bottom = top_and_bottom.replace(n, i)
    top_and_bottom = top_and_bottom.astype(int)
    return top_and_bottom


def add_by_features(df, feature_name, new_feat_name):
    for i in range(5):
        for j in range(1, 6):
            if j > i:
                df[f"{new_feat_name}{j}_by_{i}"] = df[
                    f"{j}{feature_name}"] / df[f"{i}{feature_name}"]
    return df


def multi_weighted_logloss(y_true, y_preds, classes, class_weights):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array(
        [class_weights[k] for k in sorted(class_weights.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = -np.sum(y_w) / np.sum(class_arr)
    return loss


def lgbm_multi_weighted_logloss(y_true, y_preds):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weights = {
        6: 1,
        15: 2,
        16: 1,
        42: 1,
        52: 1,
        53: 1,
        62: 1,
        64: 2,
        65: 1,
        67: 1,
        88: 1,
        90: 1,
        92: 1,
        95: 1
    }

    loss = multi_weighted_logloss(y_true, y_preds, classes, class_weights)
    return 'wloss', loss, False


def save_importances(importances_):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    return importances_


def lgbm_modeling_cross_validation(params,
                                   full_train,
                                   y,
                                   classes,
                                   class_weights,
                                   nr_fold=5,
                                   random_state=1):

    # Compute weights
    w = y.value_counts()
    weights = {i: np.sum(w) / w[i] for i in w.index}

    clfs = []
    importances = pd.DataFrame()
    folds = StratifiedKFold(
        n_splits=nr_fold, shuffle=True, random_state=random_state)

    oof_preds = np.zeros((len(full_train), np.unique(y).shape[0]))
    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full_train.iloc[val_], y.iloc[val_]

        sm = SMOTE(k_neighbors=7, n_jobs=16, random_state=42)
        trn_x, trn_y = sm.fit_resample(trn_x, trn_y)
        trn_x = pd.DataFrame(trn_x, columns=full_train.columns)
        trn_y = pd.Series(trn_y)

        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            trn_x,
            trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=lgbm_multi_weighted_logloss,
            verbose=100,
            early_stopping_rounds=100,
            sample_weight=trn_y.map(weights))
        clfs.append(clf)

        oof_preds[val_, :] = clf.predict_proba(
            val_x, num_iteration=clf.best_iteration_)
        print('no {}-fold loss: {}'.format(
            fold_ + 1,
            multi_weighted_logloss(val_y, oof_preds[val_, :], classes,
                                   class_weights)))

        imp_df = pd.DataFrame({
            'feature': full_train.columns,
            'gain': clf.feature_importances_,
            'fold': [fold_ + 1] * len(full_train.columns),
        })
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

    score = multi_weighted_logloss(
        y_true=y,
        y_preds=oof_preds,
        classes=classes,
        class_weights=class_weights)
    print('MULTI WEIGHTED LOG LOSS: {:.5f}'.format(score))
    df_importances = save_importances(importances_=importances)
    df_importances.to_csv('lgbm_importances.csv', index=False)

    return clfs, score


def gen_unknown_simple(series):
    """
    from https://www.kaggle.com/c/PLAsTiCC-2018/discussion/72104#426782
    """
    return (0.5 + 0.5 * series.median() + 0.25 * series.mean() -
            0.5 * series.max()**3) / 2.


def gen_unknown_orig(series):
    """
    from https://www.kaggle.com/c/PLAsTiCC-2018/discussion/72104
    """
    data_median = series.median()
    data_mean = series.mean()
    data_max = series.max()
    return ((((((data_median) + (((data_mean) / 2.0))) / 2.0)) + (((((1.0) - ((
        (data_max) * (((data_max) * (data_max))))))) / 2.0))) / 2.0)


def predict_chunk(df_, clfs_, meta_, features, featurize_configs, train_mean,
                  scaler, i_c):

    # 特徴量
    full_test = featurize(df_, meta_, featurize_configs['aggs'],
                          featurize_configs['fcp'])
    # 差分特徴量
    diff_test = make_diff_feature(df_)
    diff_full_test = diff_featurize(diff_test, meta_,
                                    featurize_configs['diff_aggs'],
                                    featurize_configs['fcp'])

    # 特徴量を結合
    full_test = pd.merge(full_test, diff_full_test, on='object_id')

    # ランク特徴量
    full_test = add_by_features(
        full_test, "__fft_coefficient__coeff_0__attr_\"abs\"_norm",
        "flux_norm_fft_")
    full_test = add_by_features(full_test, "__abs_energy_norm", "abs_energy_")
    full_test = add_by_features(full_test, "__flux_diff", "flux_diff_")
    abs_energy = add_rank_bottom_and_top(full_test, "__abs_energy_norm")
    flux_diff = add_rank_bottom_and_top(full_test, "__flux_diff")
    flux_dif3 = add_rank_bottom_and_top(full_test, "__flux_dif3")

    full_test = pd.merge(full_test, abs_energy, how="left", on="object_id")
    full_test = pd.merge(full_test, flux_diff, how="left", on="object_id")
    full_test = pd.merge(full_test, flux_dif3, how="left", on="object_id")

    object_id = full_test['object_id']
    if 'object_id' in full_test:
        del full_test['object_id']
        del full_test['hostgal_specz']
        del full_test['ra'], full_test['decl'], full_test['gal_l'], full_test[
            'gal_b']
        del full_test['ddf']

    # クレンジング
    full_test = full_test.replace(np.inf, np.nan)
    full_test = full_test.replace(-np.inf, np.nan)
    full_test = pd.DataFrame(
        scaler.transform(full_test),
        columns=full_test.columns,
        index=full_test.index)
    full_test.fillna(0, inplace=True)

    # データを保存
    full_test['object_id'] = object_id
    if i_c == 0:
        full_test[['object_id'] + features.tolist()].to_csv(
            './featured_test_{}.csv'.format(ver),
            header=True,
            mode='a',
            index=False)
    else:
        full_test[['object_id'] + features.tolist()].to_csv(
            './featured_test_{}.csv'.format(ver),
            header=False,
            mode='a',
            index=False)
    full_test = full_test.drop('object_id', axis=1)

    # Make predictions
    preds_ = None
    for clf in clfs_:
        if preds_ is None:
            preds_ = clf.predict_proba(full_test[features])
        else:
            preds_ += clf.predict_proba(full_test[features])

    preds_ = preds_ / len(clfs_)

    # # Compute preds_99 as the proba of class not being any of the others
    # # preds_99 = 0.1 gives 1.769
    # preds_99 = np.ones(preds_.shape[0])
    # for i in range(preds_.shape[1]):
    #     preds_99 *= (1 - preds_[:, i])

    # Create DataFrame from predictions
    preds_df_ = pd.DataFrame(
        preds_, columns=['class_{}'.format(s) for s in clfs_[0].classes_])
    preds_99 = preds_df_.apply(lambda x: gen_unknown_simple(x), axis=1)
    preds_df_['object_id'] = object_id.values
    preds_df_['class_99'] = 0.18 * preds_99 / np.mean(preds_99)
    return preds_df_


def process_test(clfs,
                 features,
                 featurize_configs,
                 scaler,
                 train_mean,
                 filename='predictions.csv',
                 chunks=5000000):
    start = time.time()

    meta_test = process_meta('../input/test_set_metadata.csv')
    # meta_test.set_index('object_id',inplace=True)
    # features=full_train.columns
    # featurize_configs={'aggs': aggs, 'diff_aggs': diff_aggs, 'fcp': fcp}
    # chunks=50000

    remain_df = None
    for i_c, df in enumerate(
            pd.read_csv(
                '../input/test_set.csv', chunksize=chunks, iterator=True)):
        # i_c = 0
        # reader = pd.read_csv('../input/test_set.csv', chunksize=chunks, iterator=True)
        # df = reader.get_chunk()

        # Check object_ids
        # I believe np.unique keeps the order of group_ids as they appear in the file
        unique_ids = np.unique(df['object_id'])

        new_remain_df = df.loc[df['object_id'] == unique_ids[-1]].copy()
        if remain_df is None:
            df = df.loc[df['object_id'].isin(unique_ids[:-1])]
        else:
            df = pd.concat(
                [remain_df, df.loc[df['object_id'].isin(unique_ids[:-1])]],
                axis=0)
        # Create remaining samples df
        remain_df = new_remain_df

        preds_df = predict_chunk(
            df_=df,
            clfs_=clfs,
            meta_=meta_test,
            features=features,
            featurize_configs=featurize_configs,
            train_mean=train_mean,
            scaler=scaler,
            i_c=i_c)

        if i_c == 0:
            preds_df.to_csv(filename, header=True, mode='a', index=False)
        else:
            preds_df.to_csv(filename, header=False, mode='a', index=False)

        del preds_df
        gc.collect()
        print(
            '{:15d} done in {:5.1f} minutes'.format(
                chunks * (i_c + 1), (time.time() - start) / 60),
            flush=True)

    # Compute last object in remain_df
    preds_df = predict_chunk(
        df_=remain_df,
        clfs_=clfs,
        meta_=meta_test,
        features=features,
        featurize_configs=featurize_configs,
        train_mean=train_mean,
        scaler=scaler,
        i_c=i_c)

    preds_df.to_csv(filename, header=False, mode='a', index=False)
    return


def save_graph_and_csv(clfs, ver, features, save=True):
    importances = []
    for clf in clfs:
        importances.append(clf.feature_importances_)
    importances = pd.Series(np.array(importances).mean(axis=0), index=features)
    importances = importances.sort_values(ascending=False)

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    importances.iloc[:40].plot(kind='barh', ax=ax)
    fig.tight_layout()

    if save:
        fig.savefig('./importance_graph_{}.png'.format(ver))
        importances.to_csv(os.path.join('./importance_{}.csv'.format(ver)))

    return importances


def get_drop_columns_by_ver1(sub_ver_list):
    drop_columns_ver1 = []
    for sub_ver in sub_ver_list:
        if sub_ver is None:
            additional_features = pd.read_csv('./additional_features_ver1.csv')
        else:
            additional_features = pd.read_csv(
                './additional_features_ver1-{}.csv'.format(sub_ver))
        drop_columns = additional_features[additional_features['is_good'] ==
                                           0]['additional_column'].tolist()
        drop_columns_ver1.extend(drop_columns)
    drop_columns_ver1 = pd.Series(drop_columns_ver1).drop_duplicates().tolist()
    return drop_columns_ver1


def main():
    # Features to compute with tsfresh library. Fft coefficient is meant to capture periodicity

    # agg features
    aggs = {
        'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'detected': ['mean'],
        'flux_ratio_sq': ['sum', 'skew'],
        'flux_by_flux_ratio_sq': ['sum', 'skew'],
    }

    # agg diff features
    diff_aggs = {
        'flux_diff': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    }

    # tsfresh features
    fcp = {
        'flux': {
            'longest_strike_above_mean': None,
            'longest_strike_below_mean': None,
            'mean_change': None,
            'mean_abs_change': None,
            'length': None,
            'mean': None,
            'maximum': None,
            'minimum': None,
            # additional
            'absolute_sum_of_changes': None,
            'autocorrelation': [{'lag': 3}],
            'binned_entropy': [{'max_bins': 10}],
            'cid_ce': [{'normalize': True}],
            'count_above_mean': None,
            'first_location_of_maximum': None,
            'first_location_of_minimum': None,
            'last_location_of_maximum': None,
            'last_location_of_minimum': None,
            'mean_second_derivative_central': None,
            'median': None,
            'ratio_beyond_r_sigma': [{'r': 2}],
            'sample_entropy': None,
            'time_reversal_asymmetry_statistic': [{'lag': 4}]
        },

        'flux_by_flux_ratio_sq': {
            'longest_strike_above_mean': None,
            'longest_strike_below_mean': None,
            # additional
            'mean_change': None,
            'mean_abs_change': None,
            'length': None,
            'mean': None,
            'maximum': None,
            'minimum': None,
            'abs_energy': None,
            'absolute_sum_of_changes': None,
            'autocorrelation': [{'lag': 3}],
            'binned_entropy': [{'max_bins': 10}],
            'cid_ce': [{'normalize': True}],
            'count_above_mean': None,
            'count_below_mean': None,
            'first_location_of_maximum': None,
            'first_location_of_minimum': None,
            'kurtosis': None,
            'longest_strike_above_mean': None,
            'longest_strike_below_mean': None,
            'mean_second_derivative_central': None,
            'median': None,
            'sample_entropy': None,
            'standard_deviation': None,
            'time_reversal_asymmetry_statistic': [{'lag': 3}]
        },

        'flux_passband': {
            'fft_coefficient': [
                    {'coeff': 0, 'attr': 'abs'},
                    {'coeff': 1, 'attr': 'abs'}
                ],
            'kurtosis': None,
            'skewness': None,
            'maximum': None,
            'mean': None,
            'minimum': None,
            # additional
            'abs_energy': None,
            'autocorrelation': [{'lag': 3}],
            'binned_entropy': [{'max_bins': 10}],
            'cid_ce': [{'normalize': True}],
            'mean_second_derivative_central': None,
            'median': None,
            'sample_entropy': None,
            'standard_deviation': None,
            'sum_values': None,
            'time_reversal_asymmetry_statistic': [{'lag': 4}]
        },

        'fcp2':  {
            "fft_coefficient": [{
                "coeff": 0,
                "attr": "abs"
            }, {
                "coeff": 1,
                "attr": "abs"
            }],
            "abs_energy": None,
            "sample_entropy": None
        },

        'mjd': {
            'maximum': None,
            'minimum': None,
            'mean_change': None,
            'mean_abs_change': None,
        },
    }

    best_params = {
            'device': 'cpu',
            'objective': 'multiclass',
            'num_class': 14,
            'boosting_type': 'gbdt',
            'n_jobs': -1,
            'max_depth': 7,
            'n_estimators': 500,
            'subsample_freq': 2,
            'subsample_for_bin': 5000,
            'min_data_per_group': 100,
            'max_cat_to_onehot': 4,
            'cat_l2': 1.0,
            'cat_smooth': 59.5,
            'max_cat_threshold': 32,
            'metric_freq': 10,
            'verbosity': -1,
            'metric': 'multi_logloss',
            'xgboost_dart_mode': False,
            'uniform_drop': False,
            'colsample_bytree': 0.5,
            'drop_rate': 0.173,
            'learning_rate': 0.0267,
            'max_drop': 5,
            'min_child_samples': 10,
            'min_child_weight': 100.0,
            'min_split_gain': 0.1,
            'num_leaves': 7,
            'reg_alpha': 0.1,
            'reg_lambda': 0.00023,
            'skip_drop': 0.44,
            'subsample': 0.75}

    # データ読込
    meta_train = process_meta('../input/training_set_metadata.csv')
    train = pd.read_csv('../input/training_set.csv')
    print('original train data shape :', train.shape)

    # 特徴量作成
    full_train = featurize(train, meta_train, aggs, fcp, n_jobs=36)
    print('full train shape :', full_train.shape)

    # 差分特徴量（ver0で作成）
    diff_train = make_diff_feature(train)
    diff_full_train = diff_featurize(diff_train, meta_train, diff_aggs, fcp, n_jobs=36)

    # 特徴量のマージ
    full_train = pd.merge(full_train, diff_full_train, on='object_id')

    # ランクの特徴量を追加
    full_train = add_by_features(
        full_train, "__fft_coefficient__coeff_0__attr_\"abs\"_norm",
        "flux_norm_fft_")
    full_train = add_by_features(full_train, "__abs_energy_norm", "abs_energy_")
    full_train = add_by_features(full_train, "__flux_diff", "flux_diff_")
    abs_energy = add_rank_bottom_and_top(full_train, "__abs_energy_norm")
    flux_diff = add_rank_bottom_and_top(full_train, "__flux_diff")
    flux_dif3 = add_rank_bottom_and_top(full_train, "__flux_dif3")

    full_train = pd.merge(full_train, abs_energy, how="left", on="object_id")
    full_train = pd.merge(full_train, flux_diff, how="left", on="object_id")
    full_train = pd.merge(full_train, flux_dif3, how="left", on="object_id")

    # 特徴量の絞り込み
    # use_features = pd.read_csv('./use_features_3-1.csv', index_col=0)
    # use_features = use_features['use_feature'].tolist()
    # full_train_raw = full_train.copy()
    # full_train = full_train_raw.copy()
    # full_train = full_train[['target']+use_features]

    if 'target' in full_train:
        y = full_train['target']
        del full_train['target']

    classes = sorted(y.unique())
    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    class_weights = {c: 1 for c in classes}
    class_weights.update({c: 2 for c in [64, 15]})
    print('Unique classes : {}, {}'.format(len(classes), classes))
    print(class_weights)

    # モデル前整形
    if 'object_id' in full_train:
        del full_train['object_id']
        # del full_train['distmod']
        del full_train['hostgal_specz']
        del full_train['ra'], full_train['decl'], full_train['gal_l'], full_train['gal_b']
        del full_train['ddf']

    train_mean = full_train.mean(axis=0).to_frame().T
    # train_mean.to_hdf('train_data.hdf5', 'data')
    pd.set_option('display.max_rows', 500)
    print(full_train.describe().T)

    full_train = full_train.replace(np.inf, np.nan)
    full_train = full_train.replace(-np.inf, np.nan)

    scl = StandardScaler()
    full_train = pd.DataFrame(scl.fit_transform(full_train), index=full_train.index, columns=full_train.columns)
    full_train.fillna(0, inplace=True)

    # クロスバリデーションの実施
    # 1回目
    eval_func = partial(lgbm_modeling_cross_validation,
                        full_train=full_train,
                        y=y,
                        classes=classes,
                        class_weights=class_weights,
                        nr_fold=7,
                        random_state=7)
    best_params.update({'n_estimators': 2000})

    clfs, score = eval_func(best_params)

    # extract importances
    importances = save_graph_and_csv(clfs, ver, full_train.columns, save=False)

    # 寄与度の低い追加特徴量を削除
    # full_train_before = full_train.copy()
    # full_train = full_train_before.copy()
    full_train = full_train[list(importances[:-250].index)]

    # 2回目
    eval_func = partial(lgbm_modeling_cross_validation,
                        full_train=full_train,
                        y=y,
                        classes=classes,
                        class_weights=class_weights,
                        nr_fold=7,
                        random_state=7)
    best_params.update({'n_estimators': 2000})

    clfs, score = eval_func(best_params)

    importances = save_graph_and_csv(clfs, ver, full_train.columns)
    pd.DataFrame({'use_feature': full_train.columns.tolist()}).to_csv('./use_features_{}.csv'.format(ver))

    # テストデータの作成
    filename = 'subm_{:.6f}_{}.csv'.format(score,
                     dt.now().strftime('%Y-%m-%d-%H-%M'))
    print('save to {}'.format(filename))
    # TEST
    # it taked about 5h
    process_test(clfs,
                 features=full_train.columns,
                 featurize_configs={'aggs': aggs, 'diff_aggs': diff_aggs, 'fcp': fcp},
                 train_mean=train_mean,
                 scaler=scl,
                 filename=filename,
                 chunks=15000000)

    z = pd.read_csv(filename)
    print("Shape BEFORE grouping: {}".format(z.shape))
    z = z.groupby('object_id').mean()
    print("Shape AFTER grouping: {}".format(z.shape))
    z.to_csv('single_{}'.format(filename), index=True)


if __name__ == '__main__':
    main()