import pandas as pd
import numpy as np


def preprocessing(data):
    from sklearn.preprocessing import OneHotEncoder,LabelEncoder
    drop_feature = ['Id','PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'Alley']
    data = data.drop(drop_feature, 1)
    data['LotFrontage'].fillna(data['LotFrontage'].mean(), inplace=True)
    data['MasVnrType'].fillna('None', inplace=True)
    data['BsmtQual'].fillna('TA', inplace=True)
    data['BsmtCond'].fillna('Gd', inplace=True)
    data['BsmtExposure'].fillna('No', inplace=True)
    data['BsmtFinType1'].fillna('Unf', inplace=True)
    data['BsmtFinType2'].fillna('Unf', inplace=True)
    data['Electrical'].fillna('SBrkr', inplace=True)
    data['GarageType'].fillna('Attchd', inplace=True)
    data['GarageYrBlt'].fillna(data['GarageYrBlt'].mean(), inplace=True)
    data['GarageFinish'].fillna('Unf', inplace=True)
    #test
    data['GarageQual'].fillna('TA', inplace=True)
    data['GarageCond'].fillna('TA', inplace=True)
    data['MasVnrArea'].fillna(data['MasVnrArea'].mean(), inplace=True)
    data['MSZoning'].fillna('RL', inplace=True)
    data['Utilities'].fillna('TA', inplace=True)
    data['Exterior1st'].fillna('VinylSd', inplace=True)
    data['Exterior2nd'].fillna('VinylSd', inplace=True)
    data['BsmtFinSF1'].fillna(data['BsmtFinSF1'].mean(), inplace=True)
    data['BsmtFinSF2'].fillna(data['BsmtFinSF1'].mean(), inplace=True)
    data['BsmtUnfSF'].fillna(data['BsmtUnfSF'].mean(), inplace=True)
    data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].mean(), inplace=True)
    data['BsmtFullBath'].fillna(data['BsmtFullBath'].mean(), inplace=True)
    data['KitchenQual'].fillna('TA', inplace=True)
    data['Functional'].fillna('Typ', inplace=True)
    data['GarageCars'].fillna('2.0', inplace=True)
    data['GarageArea'].fillna(data['GarageArea'].mean(), inplace=True)
    data['SaleType'].fillna('WD', inplace=True)
    data['BsmtHalfBath'].fillna('0.0', inplace=True)


    return data

def main():
    x_train = pd.read_csv('./data/train.csv')
    x_test = pd.read_csv('./data/test.csv')
    data = preprocessing(x_train)
    x_test = preprocessing(x_test)
    # print(x_test.count()[x_test.count()<1459])
    # print(x_test['BsmtHalfBath'].value_counts())
    from sklearn.model_selection import train_test_split
    x_train = data.drop('SalePrice', 1)
    y_train = data['SalePrice']
    from sklearn.feature_extraction import DictVectorizer
    dict_vec = DictVectorizer(sparse=False)
    x_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
    x_test = dict_vec.transform(x_test.to_dict(orient='record'))
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    sd = StandardScaler()
    x_train = sd.fit_transform(x_train)
    x_test = sd.transform(x_test)
    pca = PCA(n_components=3)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    y_predict = lr.predict(x_test)
    from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
    dnn_submission = pd.DataFrame({'Id':range(1461,2920),'SalePrice': y_predict})
    dnn_submission.to_csv('./data/lr_hourse.csv',index = False)


if __name__ == '__main__':
    main()

