#Importing Dataset and Libraries
#!pip install pydotplus
#!pip install skompiler
#!pip install astor
#!pip install joblib
#!pip install graphviz

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns


from scipy import stats
from scipy.stats import norm, skew

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

import os
import warnings
warnings.filterwarnings('ignore')


# Any results you write to the current directory are saved as output.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


warnings.simplefilter(action='ignore', category=Warning)


def load_train():
    data = pd.read_csv("/Users/mehmetkarakas/Desktop/yazılım/VBO/house_price/train.csv")
    return data

train = load_train()

def load_test():
    data = pd.read_csv("/Users/mehmetkarakas/Desktop/yazılım/VBO/house_price/test.csv")
    return data

test = load_test()

train_ids=train['Id']
test_ids=test['Id']

#Id is not important variable;therefore, we delete this variable.

train.drop('Id',axis=1, inplace=True)
test.drop('Id',axis=1, inplace=True)

train.columns.values

#Examine the data.

def check_df(dataframe, head=5):
    print("############### shape #############")
    print(dataframe.shape)
    print("############### types #############")
    print(dataframe.dtypes)
    print("############### head #############")
    print(dataframe.head())
    print("############### tail #############")
    print(dataframe.tail())
    print("############### NA #############")
    print(dataframe.isnull().sum())
    print("############### Quantiles #############")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(train)

#Missing Values
def show_missing_values(function_data):
    number_of_sample = function_data.shape[0]
    check_isnull = function_data.isnull().sum()

    check_isnull = check_isnull[check_isnull != 0].sort_values(ascending=False)

    if check_isnull.shape[0] == 0:
        print("No missing values")
        print(check_isnull)
    else:
        print(check_isnull)
        f, ax = plt.subplots(figsize=(15, 6))
        plt.xticks(rotation='90')
        sns.barplot(x=check_isnull.index, y=check_isnull)
        plt.title("The number of missing values")

show_missing_values(train)

#Correlation
#correlation among all variables.
corr=train.corr().abs()
n_most_correlated=12
#Correlation with SalePrice
most_correlated_feature=corr['SalePrice'].sort_values(ascending=False)[:n_most_correlated].drop('SalePrice')
# The hightes correlation info.
most_correlated_feature_name=most_correlated_feature.index.values


# The hightes correlation variables
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=most_correlated_feature_name, y=most_correlated_feature)
plt.title("SalePrice ile en fazla korelasyona sahip özellikler")


#Correlation with SalePrice

def draw_scatter_pairs(data,cols=4, rows=3):
    feature_names=data.columns.values

    counter=0
    fig, axarr = plt.subplots(rows,cols,figsize=(22,16))
    for i in range(rows):
        for j in range(cols):
            if counter>=len(feature_names):
                break

            name=feature_names[counter]
            axarr[i][j].scatter(x = data[name], y = data['SalePrice'])
            axarr[i][j].set(xlabel=name, ylabel='SalePrice')

            counter+=1


    plt.show()


feature_names =list(most_correlated_feature_name) + ['SalePrice']
draw_scatter_pairs(train[feature_names], rows=4, cols=3)

#Combine train and test dataset
ntrain = train.shape[0]
ntest = test.shape[0]

y_train=train['SalePrice']
X_train=train.drop('SalePrice', axis='columns')


datasets=pd.concat((X_train, test),axis='index')

print(datasets.shape)

show_missing_values(datasets)

# Solving missing problems

staretegies={}
staretegies['PoolQC']='None'
staretegies['MiscFeature']='None'
staretegies['Alley']='None'
staretegies['Fence']='None'
staretegies['FireplaceQu']='None'

staretegies['LotFrontage']='LotFrontage'

staretegies['GarageType']='None'
staretegies['GarageFinish']='None'
staretegies['GarageQual']='None'
staretegies['GarageCond']='None'

staretegies['GarageYrBlt']='Zero'
staretegies['GarageArea']='Zero'
staretegies['GarageCars']='Zero'

staretegies['BsmtFinSF1']='Zero'
staretegies['BsmtFinSF2']='Zero'
staretegies['BsmtUnfSF']='Zero'
staretegies['TotalBsmtSF']='Zero'
staretegies['BsmtFullBath']='Zero'
staretegies['BsmtHalfBath']='Zero'

staretegies['BsmtQual']='None'
staretegies['BsmtCond']='None'
staretegies['BsmtExposure']='None'
staretegies['BsmtFinType1']='None'
staretegies['BsmtFinType2']='None'

staretegies['MasVnrType']='None'
staretegies['MasVnrArea']='Zero'

staretegies['MSZoning']='Mode'

staretegies['Utilities']='Drop'
staretegies['Functional']='Functional'

staretegies['Electrical']='Mode'
staretegies['KitchenQual']='Mode'
staretegies['Exterior1st']='Mode'
staretegies['Exterior2nd']='Mode'
staretegies['SaleType']='Mode'

staretegies['MSSubClass']='None'


def fill_missing_values(fill_data, mystaretegies):
    for column, strategy in mystaretegies.items():
        if strategy == 'None':
            fill_data[column] = fill_data[column].fillna('None')
        elif strategy == 'Zero':
            fill_data[column] = fill_data[column].fillna(0)
        elif strategy == 'Mode':
            fill_data[column] = fill_data[column].fillna(fill_data[column].mode()[0])
        elif strategy == 'LotFrontage':
            # temp=fill_data.groupby("Neighborhood")
            fill_data[column] = fill_data.groupby("Neighborhood")["LotFrontage"].transform(
                lambda x: x.fillna(x.median()))
        elif strategy == 'Drop':
            fill_data = fill_data.drop([column], axis=1)
        elif strategy == 'Functional':
            fill_data[column] = fill_data[column].fillna('Typ')

    return fill_data

datasets_no_missing=fill_missing_values(datasets, staretegies)

show_missing_values(datasets_no_missing)
print(datasets_no_missing.shape)

#Encoding

for name in ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']:
    datasets_no_missing[name]= datasets_no_missing[name].astype(str)

col_names=datasets_no_missing.columns.values
col_types=datasets_no_missing.dtypes
object_cols=[]
numeric_cols=[]
for col_name, col_type in zip(col_names, col_types):
    if col_type=='object':
        object_cols.append(col_name)
    else:
        numeric_cols.append(col_name)
print("Strings:")
print(object_cols)
print("\nNumerics:")
print(numeric_cols)

label_encoder_col_names = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')

for col_name in label_encoder_col_names:
    labelEncoder=LabelEncoder()
    labelEncoder.fit(datasets_no_missing[col_name].values)
    datasets_no_missing[col_name]=labelEncoder.transform(datasets_no_missing[col_name].values)

print(datasets_no_missing.shape)

datasets_no_missing_dummies=pd.get_dummies(datasets_no_missing)
print(datasets_no_missing_dummies.shape)

#Feature Important

from yellowbrick.features import Rank1D

X_yellow = datasets_no_missing_dummies[:ntrain]
nfeature_name = datasets_no_missing_dummies.columns.values[:-1]
rank1D = Rank1D(features=nfeature_name, algorithm="shapiro")
rank1D.fit(X_yellow[nfeature_name], y_train)
rank1D.transform(X_yellow[nfeature_name])

# rank1D.poof()
print("estimating important")


df=pd.DataFrame()
df['feature_name']=nfeature_name
df['ranks']=rank1D.ranks_


df.sort_values(by=['ranks'],ascending=False, inplace=True)
df.set_index('feature_name', inplace=True)
df.head()

fig, ax=plt.subplots(1, figsize=(12,20))
df[:30].plot.barh(ax=ax)

n=30
#We took the most important variables.
n_most_important=df.index.values[:n]
print(n_most_important)

#Standartization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(datasets_no_missing_dummies[n_most_important])

#The most important variables
datasets_no_missing_dummies_scaled=scaler.transform(datasets_no_missing_dummies[n_most_important])

#Seperating train and test dataset
#Train
preprocessed_train = datasets_no_missing_dummies_scaled[:ntrain]

#Test
preprocessed_test = datasets_no_missing_dummies_scaled[ntrain:]

#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(preprocessed_train)
    rmse= np.sqrt(-cross_val_score(model, preprocessed_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

#Parameters
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

score = rmsle_cv(model_lgb)
print("LGMRboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_xgb)
print("XGBRboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

y_preds= model_xgb

# Create csv file
model_xgb.fit(preprocessed_train, y_train)
xgb_train_pred = model_xgb.predict(preprocessed_train)
xgb_pred = model_xgb.predict(preprocessed_test)

submision = pd.DataFrame()
submision['Id'] = test_ids
submision['SalePrice'] = xgb_pred

submision.to_csv('xgb_submission.csv',index=False)