import json
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

with open("names.json", "r", encoding='utf-8') as f:
    d = json.load(f)

train = pd.read_csv('data/train_data.csv')
train = train.rename(columns=d)

#-----------------------
# 前処理
#-----------------------
# 外れ値除外
train = train[train['y'] < (90)]

# 面積
train['Area'] = train['Area'].replace('2000㎡以上', '2000')
train['Area'] = train['Area'].replace('5000㎡以上', '5000')
train['Area'] = pd.to_numeric(train['Area'], errors='coerce')

# 最寄駅：距離（分）には数値以外の値が含まれているので、適当な値に置換する
train['TimeToNearestStation'] = train['TimeToNearestStation'].replace('30分?60分', '45')
train['TimeToNearestStation'] = train['TimeToNearestStation'].replace('1H?1H30', '75')
train['TimeToNearestStation'] = train['TimeToNearestStation'].replace('1H30?2H', '105')
train['TimeToNearestStation'] = train['TimeToNearestStation'].replace('2H', '120')
# pd.to_numeric() : 数値変換。
# errors='coerce' : 例外データ部分をNANで返し、他の行は数値変換
# 最寄駅：距離（分）
train['TimeToNearestStation'] = pd.to_numeric(train['TimeToNearestStation'], errors='coerce')

# 建築年を西暦に変換
train['BuildingYear'] = train['BuildingYear'].dropna()
train['BuildingYear'] = train['BuildingYear'].replace('戦前', '昭和20年')
# 年号
train['Era'] = train['BuildingYear'].str[:2]
# fillna(0)：欠損値を0に置き換える
# 和暦年数
train['YearJp'] = train['BuildingYear'].str[2:].str.strip('年').fillna(0).astype(int)
# 建築年(西暦)
train.loc[train['Era'] == '昭和', 'BuildingAD'] = train['YearJp'] + 1925
train.loc[train['Era'] == '平成', 'BuildingAD'] = train['YearJp'] + 1988

for column in ['Type','NearestStation','FloorPlan','CityPlanning','Structure', 'Direction', 'Classification', 'Municipality']:
    labels, uniques = pd.factorize(train[column])
    train[column] = labels

#------------------------
# 予測モデルの作成、学習
#------------------------
# 使用する説明変数・目的変数を代入
model_input = train[['Type', 'TimeToNearestStation','FloorPlan', 'FloorAreaRatio', 'CityPlanning', 'BuildingAD','Structure','Direction','Area', 'Classification', 'Breadth', 'CoverageRatio', 'Municipality',  'y']]

# 説明変数・目的変数についてnull値を含むレコードを除外
# how='any'：欠損値を一つでも含む行を除外
model_input = model_input.dropna(how='any', axis=0)

# 目的変数と説明変数を代入
X = model_input[['Type', 'TimeToNearestStation','FloorPlan', 'FloorAreaRatio', 'CityPlanning', 'BuildingAD','Structure','Direction', 'Area', 'Classification', 'Breadth', 'CoverageRatio','Municipality']]
y = model_input['y']

# カテゴリ変数
categorical_features = [['Type','NearestStation','FloorPlan','CityPlanning','Structure', 'Direction', 'Classification', 'Municipality']]


# train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.33, random_state=0)
cv = KFold(n_splits=5, shuffle=True)
for i, (train_index, valid_index) in enumerate(cv.split(X, y)):
    train_x = X.iloc[train_index]
    valid_x = X.iloc[valid_index]
    train_y = y.iloc[train_index]
    valid_y = y.iloc[valid_index]


# lgb_train = lgb.Dataset(train_x, train_y, categorical_feature=categorical_features)
# lgb_eval = lgb.Dataset(valid_x, valid_y,  categorical_feature=categorical_features)
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(valid_x, valid_y)

    lgbm_params = {'objective': 'regression',
                'metric':{'rmse'}}

    gbm = lgb.train(params=lgbm_params,
                train_set=lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                    num_boost_round=10000,
                early_stopping_rounds=100,
                verbose_eval=50)

# predicted = gbm.predict(valid_x)
# print(np.sqrt(mean_squared_error(valid_y, predicted)))