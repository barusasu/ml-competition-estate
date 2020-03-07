import json
import pandas as pd
import lightgbm as lgb
import numpy as np
import category_encoders as ce
# import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

with open("names.json", "r", encoding='utf-8') as f:
    d = json.load(f)

train = pd.read_csv('data/train_data.csv')
train = train.rename(columns=d)
test = pd.read_csv("data/test_data.csv")
test = test.rename(columns=d)

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

# 四半期
train['Quarter'] = train['Period'].str[6:7].astype(int)

# 間取り
train['L'] = train['FloorPlan'].map(lambda x: 1 if 'Ｌ' in str(x) else 0)
train['D'] = train['FloorPlan'].map(lambda x: 1 if 'Ｄ' in str(x) else 0)
train['K'] = train['FloorPlan'].map(lambda x: 1 if 'Ｋ' in str(x) else 0)
train['S'] = train['FloorPlan'].map(lambda x: 1 if 'Ｓ' in str(x) else 0)
train['R'] = train['FloorPlan'].map(lambda x: 1 if 'Ｒ' in str(x) else 0)
train['Maisonette'] = train['FloorPlan'].map(lambda x: 1 if 'メゾネット' in str(x) else 0)
train['OpenFloor'] = train['FloorPlan'].map(lambda x: 1 if 'オープンフロア' in str(x) else 0)
train['Studio'] = train['FloorPlan'].map(lambda x: 1 if 'スタジオ' in str(x) else 0)


Label_Enc_list = ['Type','NearestStation','FloorPlan','CityPlanning','Structure',
                 'Direction', 'Classification', 'Municipality', 'Region', 'Remarks', 'Renovation']
ce_oe = ce.OrdinalEncoder(cols=Label_Enc_list,handle_unknown='impute')
# 文字を序数に変換
train = ce_oe.fit_transform(train)
# 値を1の始まりから0の始まりにする
for i in Label_Enc_list:
    train[i] = train[i] - 1
# intに変換
for i in Label_Enc_list:
    train[i] = train[i].astype("int")

#------------------------
# 予測モデルの作成、学習
#------------------------
# 目的変数と説明変数を代入
X = train[['TimeToNearestStation','FloorAreaRatio', 'CityPlanning',
                 'BuildingAD','Structure','Direction', 'Area', 'Classification', 'Breadth',
                  'CoverageRatio','Municipality', 'Quarter', 'Region', 'Remarks', 'Renovation',
                  'L', 'D', 'K', 'OpenFloor']]
y = train['y']

# カテゴリ変数
categorical_features = ['CityPlanning','Structure', 'Direction',
                         'Classification', 'Municipality', 'Region', 'Remarks', 'Renovation']

cv = KFold(n_splits=3, shuffle=True, random_state=123)
for i, (train_index, valid_index) in enumerate(cv.split(X, y)):
    train_x = X.iloc[train_index]
    valid_x = X.iloc[valid_index]
    train_y = y.iloc[train_index]
    valid_y = y.iloc[valid_index]

    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(valid_x, valid_y)

    lgbm_params = {'objective': 'mean_squared_error',
                'metric':{'rmse'},}

    gbm = lgb.train(params=lgbm_params,
                train_set=lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                    num_boost_round=10000,
                early_stopping_rounds=100,
                verbose_eval=50,
                categorical_feature=categorical_features,)



#-------------------------
# テストデータに対する出力
#-------------------------
test['Area'] = test['Area'].replace('2000㎡以上', '2000')
test['Area'] = test['Area'].replace('5000㎡以上', '5000')
test['Area'] = pd.to_numeric(train['Area'], errors='coerce')

# 最寄駅：距離（分）には数値以外の値が含まれているので、適当な値に置換する
test['TimeToNearestStation'] = test['TimeToNearestStation'].replace('30分?60分', '45')
test['TimeToNearestStation'] = test['TimeToNearestStation'].replace('1H?1H30', '75')
test['TimeToNearestStation'] = test['TimeToNearestStation'].replace('1H30?2H', '105')
test['TimeToNearestStation'] = test['TimeToNearestStation'].replace('2H', '120')
# pd.to_numeric() : 数値変換。
# errors='coerce' : 例外データ部分をNANで返し、他の行は数値変換
# 最寄駅：距離（分）
test['TimeToNearestStation'] = pd.to_numeric(test['TimeToNearestStation'], errors='coerce')

# 建築年を西暦に変換
test['BuildingYear'] = test['BuildingYear'].dropna()
test['BuildingYear'] = test['BuildingYear'].replace('戦前', '昭和20年')
# 年号
test['Era'] = test['BuildingYear'].str[:2]
# fillna(0)：欠損値を0に置き換える
# 和暦年数
test['YearJp'] = test['BuildingYear'].str[2:].str.strip('年').fillna(0).astype(int)
# 建築年(西暦)
test.loc[test['Era'] == '昭和', 'BuildingAD'] = test['YearJp'] + 1925
test.loc[test['Era'] == '平成', 'BuildingAD'] = test['YearJp'] + 1988

# 四半期
test['Quarter'] = test['Period'].str[6:7].astype(int)

# 間取り
test['L'] = test['FloorPlan'].map(lambda x: 1 if 'Ｌ' in str(x) else 0)
test['D'] = test['FloorPlan'].map(lambda x: 1 if 'Ｄ' in str(x) else 0)
test['K'] = test['FloorPlan'].map(lambda x: 1 if 'Ｋ' in str(x) else 0)
test['S'] = test['FloorPlan'].map(lambda x: 1 if 'Ｓ' in str(x) else 0)
test['R'] = test['FloorPlan'].map(lambda x: 1 if 'Ｒ' in str(x) else 0)
test['Maisonette'] = test['FloorPlan'].map(lambda x: 1 if 'メゾネット' in str(x) else 0)
test['OpenFloor'] = test['FloorPlan'].map(lambda x: 1 if 'オープンフロア' in str(x) else 0)
test['Studio'] = test['FloorPlan'].map(lambda x: 1 if 'スタジオ' in str(x) else 0)

Label_Enc_list = ['Type','NearestStation','FloorPlan','CityPlanning','Structure',
                 'Direction', 'Classification', 'Municipality', 'Region', 'Remarks', 'Renovation']
ce_oe = ce.OrdinalEncoder(cols=Label_Enc_list,handle_unknown='impute')
# 文字を序数に変換
test = ce_oe.fit_transform(test)
# 値を1の始まりから0の始まりにする
for i in Label_Enc_list:
    test[i] = test[i] - 1
# intに変換
for i in Label_Enc_list:
    test[i] = test[i].astype("int")

# 使用する説明変数・目的変数を代入
X_test = test[['TimeToNearestStation', 'FloorAreaRatio', 'CityPlanning',
                 'BuildingAD','Structure','Direction', 'Area', 'Classification', 'Breadth',
                  'CoverageRatio','Municipality', 'Quarter', 'Region', 'Remarks', 'Renovation',
                  'L', 'D', 'K', 'OpenFloor']]

test_predicted = gbm.predict(X_test)

submit_df = pd.DataFrame({'y': test_predicted})
submit_df.index.name = 'id'
submit_df.index = submit_df.index + 1
submit_df.to_csv('submission.csv')

# importanceを表示する
importance = pd.DataFrame(gbm.feature_importance(), index=X.columns, columns=['importance'])
print(importance)