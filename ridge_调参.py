import pandas as pd
import os
import gc
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time
import warnings
from sklearn.ensemble import RandomForestRegressor as rf
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
y_train = train_df['temperature'].values - train_df['outdoorTemp'].values

for col in features:
    ss = MinMaxScaler()
    ss.fit(np.vstack([x_train[[col]].values, x_test[[col]].values]))
    x_train[col] = ss.transform(x_train[[col]].values).flatten()
    x_test[col] = ss.transform(x_test[[col]].values).flatten()

from sklearn.metrics import mean_squared_error
nums = int(x_train.shape[0] * 0.80)
trn_x, trn_y, val_x, val_y = x_train[:nums], y_train[:nums], x_train[nums:], y_train[nums:]

ridge = Ridge(alpha=100, random_state=0)
ridge.fit(trn_x, trn_y)
val_pred = ridge.predict(val_x)
test_pred = ridge.predict(x_test)
print(mean_squared_error(val_y, val_pred))