import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from merger.create_full_train import train_data
from loss.loss_func import lgb_multi_weighted_logloss
from loss.loss_func import multi_weighted_logloss
from plotting import save_importances, save_cm

if __name__ == "__main__":
    data_dir = "/Users/hidehisa/.kaggle/competitions/plasticc"
    train = pd.read_csv(data_dir + "/train_with_cluster.csv")
    meta = pd.read_csv(data_dir + "/training_set_metadata.csv")

    full, y, classes, class_weight, oof_df = train_data(train, meta)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    clfs = []
    importances = pd.DataFrame()

    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 14,
        'metric': 'multi_logloss',
        'learning_rate': 0.03,
        'subsample': .9,
        'colsample_bytree': 0.5,
        'reg_alpha': .01,
        'reg_lambda': .01,
        'min_split_gain': 0.01,
        'min_child_weight': 10,
        'n_estimators': 1000,
        'silent': -1,
        'verbose': -1,
        'max_depth': 3
    }

    # Compute weights
    w = y.value_counts()
    weights = {i: np.sum(w) / w[i] for i in w.index}
    oof_preds = np.zeros((len(full), np.unique(y).shape[0]))

    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        trn_x, trn_y = full.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full.iloc[val_], y.iloc[val_]

        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(
            trn_x,
            trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=lgb_multi_weighted_logloss,
            verbose=100,
            early_stopping_rounds=50,
            sample_weight=trn_y.map(weights))
        oof_preds[val_, :] = clf.predict_proba(
            val_x, num_iteration=clf.best_iteration_)
        print(multi_weighted_logloss(val_y, oof_preds[val_, :]))

        imp_df = pd.DataFrame()
        imp_df['feature'] = full.columns
        imp_df['gain'] = clf.feature_importances_
        imp_df['fold'] = fold_ + 1
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

        clfs.append(clf)

    print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(
        y_true=y, y_preds=oof_preds))
    save_importances(importances_=importances)
    save_cm(y, oof_preds, data_dir + "/sample_submission.csv")
