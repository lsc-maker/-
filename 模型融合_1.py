import pandas as pd
import os
import gc
import lightgbm as lgb
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time
import warnings
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

train_df = pd.read_csv(r'D:\比赛\科大讯飞\温室温度预测挑战赛_温室温度预测赛初赛数据\train/train.csv')
test_df = pd.read_csv(r'D:\比赛\科大讯飞\温室温度预测挑战赛_温室温度预测赛初赛数据\test/test.csv')
sub = pd.DataFrame(test_df['time'])

train_df = train_df[train_df['temperature'].notnull()]
train_df = train_df.fillna(method='bfill')
test_df = test_df.fillna(method='bfill')

train_df.columns = ['time', 'year', 'month', 'day', 'hour', 'min', 'sec', 'outdoorTemp', 'outdoorHum', 'outdoorAtmo',
                    'indoorHum', 'indoorAtmo', 'temperature']
test_df.columns = ['time', 'year', 'month', 'day', 'hour', 'min', 'sec', 'outdoorTemp', 'outdoorHum', 'outdoorAtmo',
                   'indoorHum', 'indoorAtmo']

data_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# 基本聚合特征
group_feats = []
for f in tqdm(['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']):
    data_df['MDH_{}_medi'.format(f)] = data_df.groupby(['month', 'day', 'hour'])[f].transform('median')
    data_df['MDH_{}_mean'.format(f)] = data_df.groupby(['month', 'day', 'hour'])[f].transform('mean')
    data_df['MDH_{}_max'.format(f)] = data_df.groupby(['month', 'day', 'hour'])[f].transform('max')
    data_df['MDH_{}_min'.format(f)] = data_df.groupby(['month', 'day', 'hour'])[f].transform('min')
    data_df['MDH_{}_std'.format(f)] = data_df.groupby(['month', 'day', 'hour'])[f].transform('std')

    group_feats.append('MDH_{}_medi'.format(f))
    group_feats.append('MDH_{}_mean'.format(f))

# 基本交叉特征
for f1 in tqdm(['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo'] + group_feats):

    for f2 in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo'] + group_feats:
        if f1 != f2:
            colname = '{}_{}_ratio'.format(f1, f2)
            data_df[colname] = data_df[f1].values / data_df[f2].values

data_df = data_df.fillna(method='bfill')

# 历史信息提取
data_df['dt'] = data_df['day'].values + (data_df['month'].values - 3) * 31

for f in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo', 'temperature']:
    tmp_df = pd.DataFrame()
    for t in tqdm(range(15, 45)):
        tmp = data_df[data_df['dt'] < t].groupby(['hour'])[f].agg({'mean'}).reset_index()
        tmp.columns = ['hour', 'hit_{}_mean'.format(f)]
        tmp['dt'] = t
        tmp_df = tmp_df.append(tmp)

    data_df = data_df.merge(tmp_df, on=['dt', 'hour'], how='left')

data_df = data_df.fillna(method='bfill')

# 离散化
for f in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']:
    data_df[f + '_20_bin'] = pd.cut(data_df[f], 20, duplicates='drop').apply(lambda x: x.left).astype(int)
    data_df[f + '_50_bin'] = pd.cut(data_df[f], 50, duplicates='drop').apply(lambda x: x.left).astype(int)
    data_df[f + '_100_bin'] = pd.cut(data_df[f], 100, duplicates='drop').apply(lambda x: x.left).astype(int)
    data_df[f + '_200_bin'] = pd.cut(data_df[f], 200, duplicates='drop').apply(lambda x: x.left).astype(int)

for f1 in tqdm(
        ['outdoorTemp_20_bin', 'outdoorHum_20_bin', 'outdoorAtmo_20_bin', 'indoorHum_20_bin', 'indoorAtmo_20_bin']):
    for f2 in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']:
        data_df['{}_{}_medi'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('min')

for f1 in tqdm(
        ['outdoorTemp_50_bin', 'outdoorHum_50_bin', 'outdoorAtmo_50_bin', 'indoorHum_50_bin', 'indoorAtmo_50_bin']):
    for f2 in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']:
        data_df['{}_{}_medi'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('min')

for f1 in tqdm(['outdoorTemp_100_bin', 'outdoorHum_100_bin', 'outdoorAtmo_100_bin', 'indoorHum_100_bin',
                'indoorAtmo_100_bin']):
    for f2 in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']:
        data_df['{}_{}_medi'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('min')

for f1 in tqdm(['outdoorTemp_200_bin', 'outdoorHum_200_bin', 'outdoorAtmo_200_bin', 'indoorHum_200_bin',
                'indoorAtmo_200_bin']):
    for f2 in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']:
        data_df['{}_{}_medi'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format
        (f1, f2)] = data_df.groupby([f1])[f2].transform('min')

drop_columns = ["time", "year", "sec", "temperature"]

train_count = train_df.shape[0]
train_df = data_df[:train_count].copy().reset_index(drop=True)
test_df = data_df[train_count:].copy().reset_index(drop=True)


features = train_df[:1].drop(drop_columns, axis=1).columns
x_train = train_df[features]
x_test = test_df[features]
# x_val = x_train[x_train.shape[0] * 0.80:]

y_train = train_df['temperature'].values - train_df['outdoorTemp'].values

nums = int(x_train.shape[0] * 0.80)
trn_x, trn_y, val_x, val_y = x_train[:nums], y_train[:nums], x_train[nums:], y_train[nums:]
####################lgb
train_matrix = lgb.Dataset(trn_x, label=trn_y)
valid_matrix = lgb.Dataset(val_x, label=val_y)
data_matrix = lgb.Dataset(x_train, label=y_train)

params = {
            'boosting_type': 'gbdt', 'objective': 'mse', 'min_child_weight': 5, 'num_leaves': 2 ** 8,
            'feature_fraction': 0.5, 'bagging_fraction': 0.5, 'bagging_freq': 1, 'learning_rate': 0.001, 'seed': 2020
        }

model_lgb = lgb.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], verbose_eval=500,
                          early_stopping_rounds=1000)

####################xgb
train_matrix = xgb.DMatrix(trn_x, label=trn_y, missing=np.nan)
valid_matrix = xgb.DMatrix(val_x, label=val_y, missing=np.nan)
test_matrix = xgb.DMatrix(x_test, label=val_y, missing=np.nan)
params = {'booster': 'gbtree',
                  'n_estimators' : 485,
                  'eval_metric': 'mae',
                  'min_child_weight': 5,
                  'max_depth': 5,
                  'subsample': 0.8041,
                  'colsample_bytree': 0.9289,
                  'eta': 0.001,
                  'seed': 2020,
                  'nthread': 36,
                  'silent': True,
            }

watchlist = [(train_matrix, 'train'), (valid_matrix, 'eval')]

model_xgb = xgb.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=500,
                          early_stopping_rounds=1000)

####################cat
params = {'learning_rate': 0.001, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
                  'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}

model_cat = CatBoostRegressor(iterations=20000, **params)
model_cat.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                  cat_features=[], use_best_model=True, verbose=500)

###################rf
params = {
            "max_depth": 9,
            "max_features": 0.4717,
            'min_samples_split': 15,
            'n_estimators': 36
        }
model_rf = RF(**params)
model_rf.fit(trn_x, trn_y)

#################SGD
params = {
            'loss': 'squared_loss',
            'penalty': 'l2',
            'alpha': 0.00001,
            'random_state': 2020,
        }
model_SGD = SGDRegressor(**params)
model_SGD.fit(trn_x, trn_y)

#############################ridge
params = {
            'alpha': 1.0,
            'random_state': 2020,
        }
model_ridge = Ridge(**params)
model_ridge.fit(trn_x, trn_y)

train_lgb_predict = model_lgb.predict(trn_x)
train_xgb_predict = model_xgb.predict(train_matrix)
train_cat_predict = model_cat.predict(trn_x)
train_rf_predict = model_rf.predict(trn_x)
train_sgd_predict = model_SGD.predict(trn_x)
train_ridge_predict = model_ridge.predict(trn_x)

Strak_X_train = pd.DataFrame()
Strak_X_train['Method_1'] = train_lgb_predict
Strak_X_train['Method_2'] = train_xgb_predict
Strak_X_train['Method_3'] = train_cat_predict
Strak_X_train['Method_4'] = train_rf_predict
Strak_X_train['Method_5'] = train_sgd_predict
Strak_X_train['Method_6'] = train_ridge_predict

val_lgb = model_lgb.predict(val_x)
val_xgb = model_xgb.predict(valid_matrix)
val_cat = model_cat.predict(val_x)
val_rf = model_rf.predict(val_x)
val_sgd = model_SGD.predict(val_x)
val_ridge = model_ridge.predict(val_x)

Strak_X_val = pd.DataFrame()
Strak_X_val['Method_1'] = val_lgb
Strak_X_val['Method_2'] = val_xgb
Strak_X_val['Method_3'] = val_cat
Strak_X_val['Method_4'] = val_rf
Strak_X_val['Method_5'] = val_sgd
Strak_X_val['Method_6'] = val_ridge

sub_lgb = model_lgb.predict(x_test)
sub_xgb = model_xgb.predict(test_matrix)
sub_cat = model_cat.predict(x_test)
sub_rf = model_rf.predict(x_test)
sub_sgd = model_SGD.predict(x_test)
sub_ridge = model_ridge.predict(x_test)

Strak_X_test = pd.DataFrame()
Strak_X_test['Method_1'] = sub_lgb
Strak_X_test['Method_2'] = sub_xgb
Strak_X_test['Method_3'] = sub_cat
Strak_X_test['Method_4'] = sub_rf
Strak_X_test['Method_5'] = sub_sgd
Strak_X_test['Method_6'] = sub_ridge

def build_model_lr(x_train,y_train):
    reg_model = LinearRegression()
    reg_model.fit(x_train,y_train)
    return reg_model
model_lr_Stacking = build_model_lr(Strak_X_train,trn_y)

## 训练集
train_pre_Stacking = model_lr_Stacking.predict(Strak_X_train)
print('MAE of Stacking-LR:',mean_absolute_error(trn_y,train_pre_Stacking))

## 验证集
val_pre_Stacking = model_lr_Stacking.predict(Strak_X_val)
print('MAE of Stacking-LR:',mean_absolute_error(val_y,val_pre_Stacking))

## 预测集
print('Predict Stacking-LR...')
subA_Stacking = model_lr_Stacking.predict(Strak_X_test)

sub["temperature"] = subA_Stacking + test_df['outdoorTemp'].values
sub.to_csv('sub4.csv', index=False)