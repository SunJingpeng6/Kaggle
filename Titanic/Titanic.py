import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn import linear_model

class Titanic():
    def __init__(self):
        # RandomForestRegressor dealing with age missing data
        self.age_model = None
        # LogisticRegression classifier to predict passengers "Survived" or not
        self.logistic_clf = None

    def fit(self, data_train):
        # preprocessing the train data
        data_train = self._preprocess(data_train)
        # classifier training
        self._logistic_clf(data_train)

    def predict(self, data_test):
        # deal with 'Fare' missing data in test data, notice 'Fare' data is full
        # in train data
        data_test = self._set_missing_fare(data_test)
        # preprocessing the test data
        data_test = self._preprocess(data_test)
        # select features used by LogisticRegression
        test_df = data_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
        # transform pandas format into numpy format
        X_test = test_df.values
        y_pred = self.logistic_clf.predict(X_test)
        # save result to file
        self._save(data_test, y_pred)
        return y_pred

    def _set_missing_fare(self, df):
        fare_df = df[['Fare']]
        known_fare = fare_df[fare_df.Fare.notnull()].values
        # predict missing data by averaging other data
        fare_pred = known_fare.mean()
        df.loc[(data_test.Fare.isnull()), 'Fare'] = fare_pred
        return df

    def _save(self, data_test, y_pred ):
        result = pd.DataFrame({"PassengerId": data_test['PassengerId'].values, 'Survived': y_pred.astype(np.int32)})
        result.to_csv('logistic_regression_predictions.csv', index=False)

    def _preprocess(self, df):
        # deal with missing age data
        df = self._set_missing_age(df)
        # deal with Cabin data
        df = self._set_Carbin_type(df)
        df = self._discrete_factored(df)
        df = self._normalized(df)
        return df

    def _set_missing_age(self, df):
        # use RandomForestRegressor to predict missing age
        age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

        known_age = age_df[age_df.Age.notnull()].values
        unknown_age = age_df[age_df.Age.isnull()].values

        # deal with test data
        if self.age_model is not None:
            age_pred = self.age_model.predict(unknown_age[:, 1:])
        # deal with train data
        else:
            X_train = known_age[:, 1:]
            y_train = known_age[:, 0]
            age_model = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)
            age_model.fit(X_train, y_train)
            self.age_model = age_model
            age_pred = age_model.predict(unknown_age[:, 1:])
        df.loc[(df.Age.isnull()), 'Age'] = age_pred
        return df

    def _set_Carbin_type(self, df):
        # a lot of Cabin data is missing, so transform it to 'yes' or 'no' based on existence
        df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'
        df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'
        return df

    def _discrete_factored(self, df):
        # 将非数值数据数值化
        dummies_Carbin = pd.get_dummies(df['Cabin'], prefix='Cabin')
        dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
        dummies_Sex = pd.get_dummies(df['Sex'], prefix='Sex')
        dummies_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')
        df = pd.concat([df, dummies_Carbin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
        df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
        return df

    def _normalized(self, df):
        # 数值型数据Age Fare变化很大，将其标准化
        scaler = preprocessing.StandardScaler()

        age_scale_param = scaler.fit(df[['Age']])
        df['Age_scaled'] = age_scale_param.transform(df[['Age']])
        df['Age'] = df['Age_scaled']

        fare_scale_param = scaler.fit(df[['Fare']])
        df['Fare_scaled'] = fare_scale_param.transform(df[['Fare']])
        df['Fare'] = df['Fare_scaled']
        return df

    def _logistic_clf(self, data_train):
        # LogisticRegression
        train_df = data_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
        train_np = train_df.values
        X_train = train_np[:, 1:]
        y_train = train_np[:, 0]
        # initializie classifier
        self.logistic_clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        # fit classifier
        self.logistic_clf.fit(X_train, y_train)

if __name__ == '__main__':
    data_train = pd.read_csv('train.csv')
    data_test = pd.read_csv('test.csv')

    clf = Titanic()
    clf.fit(data_train)
    
    y_pred = clf.predict(data_test)
    print(y_pred)
