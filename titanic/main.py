# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style('whitegrid')
# %matplotlib inline

# machine learning
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, f1_score

d=dict([("female",0),("male",1)])
def map_sex(s):
    return d.get(s,0)
def data_processing(pl=True, ae=True, sb=True, ph=True, fr=True):
    # get titanic & test csv files as a DataFrame
    titanic_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")
    # drop unnecessary columns, these columns won't be useful in analysis and prediction
    titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
    titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
    titanic_df["Fare"].fillna(titanic_df["Fare"].median(), inplace=True)
    titanic_df["Age"].fillna(titanic_df["Age"].median(), inplace=True)
    titanic_df.drop("Cabin", axis=1, inplace=True)
    # test_DATA PREPROCESSING
    test_df = test_df.drop(['PassengerId','Name', 'Ticket'], axis=1)
    test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
    test_df.drop("Cabin", axis=1, inplace=True)
    test_df["Age"].fillna(test_df["Age"].median(), inplace=True)

    # 2、得到标注
    train_label = titanic_df["Survived"]
    titanic_df = titanic_df.drop("Survived", axis=1)
    scaler_lst = [pl, ae, sb, ph, fr]
    column_lst = ["Pclass", "Age", "SibSp", \
                  "Parch", "Fare", "Work_accident"]
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            titanic_df[column_lst[i]] =MinMaxScaler().fit_transform(titanic_df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
            test_df[column_lst[i]] =MinMaxScaler().fit_transform(test_df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            titanic_df[column_lst[i]] =StandardScaler().fit_transform(titanic_df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
            test_df[column_lst[i]] =MinMaxScaler().fit_transform(test_df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
    scaler_lst=[False,True]
    column_lst = ["Sex", "Embarked"]
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            if column_lst[i] == "Sex":
                titanic_df[column_lst[i]] = [map_sex(s) for s in titanic_df["Sex"].values]
                test_df[column_lst[i]] = [map_sex(s) for s in test_df["Sex"].values]
            else:
                titanic_df[column_lst[i]] = LabelEncoder().fit_transform(titanic_df[column_lst[i]])
                test_df[column_lst[i]] = LabelEncoder().fit_transform(test_df[column_lst[i]])
            titanic_df[column_lst[i]] = MinMaxScaler().fit_transform(titanic_df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
            test_df[column_lst[i]] = MinMaxScaler().fit_transform(test_df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            titanic_df = pd.get_dummies(titanic_df, columns=[column_lst[i]])
            test_df = pd.get_dummies(test_df, columns=[column_lst[i]])
    return titanic_df,test_df,train_label

def data_modeling(features,label,test_data):
    X_train=features.values
    # Y_train = test_data.values
    Y_train = label
    models = []
    models.append(("KNN",KNeighborsClassifier(n_neighbors=3)))
    models.append(("GaussianNB",GaussianNB()))
    # models.append(("BernoulliNB",BernoulliNB()))
    # models.append(("LogisticRegression", LogisticRegression()))
    # models.append(("DecisionTreeGini", DecisionTreeClassifier()))
    # models.append(("DecisionTreeEntropy", DecisionTreeClassifier(criterion="entropy")))
    knn_clf = KNeighborsClassifier(n_neighbors=3)
    for clf_name, clf in models:
        clf.fit(X_train, Y_train)
        xy_lst = [(X_train, Y_train)]
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = clf.predict(X_part)
            print(i)
            print(clf_name, "ACC:", accuracy_score(Y_part, Y_pred))
            print(clf_name, "REC:", recall_score(Y_part, Y_pred))
            print(clf_name, "F-Score:", f1_score(Y_part, Y_pred))
            # dot_data=StringIO()
            # export_graphviz(clf,out_file=dot_data,feature_names=f_names,class_names=["NL","L"],filled=True,rounded=True,special_characters=True)
            # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
            # graph.write_pdf("dt_tree_2.pdf")
def __main__():
    titanic_df, test_df, train_label = data_processing()
    data_modeling(titanic_df,train_label, test_df )

__main__()
# preview the data
