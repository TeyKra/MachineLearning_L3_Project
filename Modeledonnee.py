import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import stats
from sklearn.datasets import make_blobs
import seaborn as sns
import sklearn.cluster as skc
import sklearn.preprocessing as skp
import csv

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor


class DataLoad():
    def __init__(self):
        self.dfX = pd.read_csv("Data_X.csv")
        self.dfY = pd.read_csv("Data_Y.csv")
        self.dfNew_X = pd.read_csv("DataNew_X (1).csv")
        self.FR_dfX = self.dfX[self.dfX['COUNTRY'] == 'FR']
        self.FR_dfY = self.dfY[self.dfX['COUNTRY'] == 'FR']
        self.DE_dfX = self.dfX[self.dfX['COUNTRY'] == 'DE']
        self.DE_dfY = self.dfY[self.dfX['COUNTRY'] == 'DE']


    def missing_values(self):
        print(self.dfX.isnull().sum())
        print(self.dfY.isnull().sum())
        print(self.dfNew_X.isnull().sum())

    def normalisation(self):
        #Normalisation des données
        pd.options.mode.chained_assignment = None
        self.FR_dfX.drop(['DE_FR_EXCHANGE','FR_NET_EXPORT','DE_NET_EXPORT'], axis=1, inplace=True)
        self.DE_dfX.drop(['DE_FR_EXCHANGE', 'FR_NET_EXPORT', 'DE_NET_EXPORT'], axis=1, inplace=True)

        self.FR_dfX.drop(['COUNTRY', 'DAY_ID'], axis=1, inplace=True)
        self.DE_dfX.drop(['COUNTRY', 'DAY_ID'], axis=1, inplace=True)

        self.FR_dfX['TARGET'] = self.dfY['TARGET']
        self.DE_dfX['TARGET'] = self.dfY['TARGET']
        self.FR_dfX.fillna(self.FR_dfX.mean(), inplace=True)
        self.DE_dfX.fillna(self.DE_dfX.mean(), inplace=True)

    def comparable_values(self):
        for i in self.FR_dfX.columns:
            if(self.FR_dfX[i].dtypes==self.FR_dfX['TARGET'].dtypes):
                print("Les valeurs sont comparables")
            else:
                print("Les valeurs ne sont pas comparables")
    def info(self):
        print(self.FR_dfX.info())
        print(self.DE_dfX.info())
        print(self.FR_dfY.info())
        print(self.DE_dfY.info())
    def describe(self):
        print(self.FR_dfX.describe())
        print(self.DE_dfX.describe())
        print(self.FR_dfY.describe())
        print(self.DE_dfY.describe())
    def verifNA(self):
        print(self.FR_dfX.isnull().sum())
        print(self.DE_dfX.isnull().sum())
        print(self.FR_dfY.isnull().sum())
        print(self.DE_dfY.isnull().sum())

    def comparaison(self):
        for i in self.FR_dfX.columns:
            print(i)
            print(self.FR_dfX[i].unique())
        for i in self.DE_dfX.columns:
            print(i)
            print(self.DE_dfX[i].unique())
        for i in self.FR_dfY.columns:
            print(i)
            print(self.FR_dfY[i].unique())
        for i in self.DE_dfY.columns:
            print(i)
            print(self.DE_dfY[i].unique())
    def showHisto(self):
        for i in self.FR_dfX.columns:
            self.FR_dfX[i].hist()
            plt.title(i)
            plt.show()

        for i in self.DE_dfX.columns:
            self.DE_dfX[i].hist()
            plt.title(i)
            plt.show()
        for i in self.FR_dfY.columns:
            self.FR_dfY[i].hist()
            plt.title(i)
            plt.show()

        for i in self.DE_dfY.columns:
            self.DE_dfY[i].hist()
            plt.title(i)
            plt.show()
    def boxplot(self):
        for i,j in enumerate(self.FR_dfX.describe().columns):
            #plt.subplot(3, 3, i+1)
            sns.boxplot(x=self.FR_dfX[j])
            plt.title('{} Boxplot'.format(j))
            plt.tight_layout()
            plt.show()
    def showCorrelation(self):
        dfco= self.FR_dfX.select_dtypes(exclude=['object'])
        dfco2 = self.DE_dfX.select_dtypes(exclude=['object'])
        print(dfco.corr())
        print(dfco2.corr())

    def shape(self):
        print(self.FR_dfX.shape)
        print(self.DE_dfX.shape)
        print(self.FR_dfY.shape)
        print(self.DE_dfY.shape)

    def DEBoxplot(self):
        for i, j in enumerate(self.DE_dfX.describe().columns):
            # plt.subplot(3, 3, i+1)
            sns.boxplot(x=self.DE_dfX[j])
            plt.title('{} Boxplot'.format(j))
            plt.tight_layout()
            plt.show()
            print("success")
    def regressionlinearDE(self):
        DEX_train, DEX_test, DEy_train, DEy_test = train_test_split(self.DE_dfX, self.DE_dfY, test_size=0.3 ,random_state=42)
        model2 = LinearRegression()
        model2.fit(DEX_train, DEy_train)
        DEy_pred = model2.predict(DEX_test)
        plt.plot(DEy_test, DEy_test, color='red')
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.show()
        rmse = np.sqrt(mean_squared_error(DEy_test, DEy_pred))
        r2 = r2_score(DEy_test, DEy_pred)
        print("RMSE")
        print(rmse)
        print("R2")
        print(r2)
        res=stats.spearmanr(DEy_test,DEy_pred)
        print(res)
    def regressionlinearFR(self):
        FRX_train, FRX_test, FRy_train, FRy_test = train_test_split(self.FR_dfX, self.FR_dfY, test_size=0.9,random_state=42)

        model = LinearRegression()
        model.fit(FRX_train, FRy_train)
        FRy_pred = model.predict(FRX_test)
        plt.plot(FRy_test, FRy_test, color='red')
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.show()
        rmse = np.sqrt(mean_squared_error(FRy_test, FRy_pred))
        r2 = r2_score(FRy_test, FRy_pred)
        print("RMSE")
        print(rmse)
        print("R2")
        print(r2)
        res = stats.spearmanr(FRy_test, FRy_pred)
        print(res)

    def RIDGEregression(self):
        ridge = Ridge()
        parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
        ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
        ridge_regressor.fit(self.DE_dfX, self.DE_dfY)
        DEX_train, DEX_test, DEy_train, DEy_test = train_test_split(self.DE_dfX, self.DE_dfY, test_size=0.3,random_state=42)
        predictions = ridge_regressor.predict(DEX_test)
        print(predictions)
        sns.displot(DEy_test - predictions)
        plt.show()
        print(ridge_regressor.best_params_)
        print(ridge_regressor.best_score_)
        rmse = np.sqrt(mean_squared_error(DEy_test, predictions))
        print("rmse",rmse)
        print("R2 score : %.2f" % r2_score(DEy_test, predictions))
        res= stats.spearmanr(DEy_test, predictions)
        print(res)
    def RIDGEregressionFR(self):
        ridge = Ridge()
        parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
        ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
        ridge_regressor.fit(self.FR_dfX, self.FR_dfY)
        FRX_train, FRX_test, FRy_train, FRy_test = train_test_split(self.FR_dfX, self.FR_dfY, test_size=0.3,random_state=42)
        predictions = ridge_regressor.predict(FRX_test)
        print(predictions)
        sns.displot(FRy_test - predictions)
        plt.show()
        print(ridge_regressor.best_params_)
        print(ridge_regressor.best_score_)
        rmse = np.sqrt(mean_squared_error(FRy_test, predictions))
        print("rmse",rmse)
        print("R2 score : %.2f" % r2_score(FRy_test, predictions))
        res= stats.spearmanr(FRy_test, predictions)
        print(res)
    def LASSOregression(self):
        lasso = Lasso()
        parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
        lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
        lasso_regressor.fit(self.DE_dfX, self.DE_dfY)
        DEX_train, DEX_test, DEy_train, DEy_test = train_test_split(self.DE_dfX, self.DE_dfY, test_size=0.3,random_state=42)
        predictions = lasso_regressor.predict(DEX_test)
        print(predictions)
        sns.displot(DEy_test - predictions)
        plt.show()
        print(lasso_regressor.best_params_)
        print(lasso_regressor.best_score_)
        rmse = np.sqrt(mean_squared_error(DEy_test, predictions))
        print("rmse",rmse)
        print("R2 score : %.2f" % r2_score(DEy_test, predictions))
        res= stats.spearmanr(DEy_test, predictions)
        print(res)
    def LASSOregressionFR(self):
        lasso = Lasso()
        parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
        lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
        lasso_regressor.fit(self.FR_dfX, self.FR_dfY)
        FRX_train, FRX_test, FRy_train, FRy_test = train_test_split(self.FR_dfX, self.FR_dfY, test_size=0.3,random_state=42)
        predictions = lasso_regressor.predict(FRX_test)
        print(predictions)
        sns.displot(FRy_test - predictions)
        plt.show()
        print(lasso_regressor.best_params_)
        print(lasso_regressor.best_score_)
        rmse = np.sqrt(mean_squared_error(FRy_test, predictions))
        print("rmse",rmse)
        print("R2 score : %.2f" % r2_score(FRy_test, predictions))
        res= stats.spearmanr(FRy_test, predictions)
        print(res)

    def DecisionTreeregression(self):
        regressor = DecisionTreeRegressor()
        regressor.fit(self.DE_dfX, self.DE_dfY)
        DEX_train, DEX_test, DEy_train, DEy_test = train_test_split(self.DE_dfX, self.DE_dfY, test_size=0.3,random_state=42)
        predictions = regressor.predict(DEX_test)
        print(predictions)
        sns.displot(DEy_test - predictions)
        plt.show()
        print(regressor.best_params_)
        print(regressor.best_score_)
        rmse = np.sqrt(mean_squared_error(DEy_test, predictions))
        print("rmse",rmse)
        print("R2 score : %.2f" % r2_score(DEy_test, predictions))
        res= stats.spearmanr(DEy_test, predictions)
        print(res)

        plt.show()
    def DecisionTreeregressionFR(self):
        regressor = DecisionTreeRegressor()
        regressor.fit(self.FR_dfX, self.FR_dfY)
        FRX_train, FRX_test, FRy_train, FRy_test = train_test_split(self.FR_dfX, self.FR_dfY, test_size=0.3,random_state=42)
        predictions = regressor.predict(FRX_test)
        print(predictions)
        sns.displot(FRy_test - predictions)
        plt.show()
        print(regressor.best_params_)
        print(regressor.best_score_)
        rmse = np.sqrt(mean_squared_error(FRy_test, predictions))
        print("rmse",rmse)
        print("R2 score : %.2f" % r2_score(FRy_test, predictions))
        res= stats.spearmanr(FRy_test, predictions)
        print(res)
    def RandomForestregression(self):
        regressor = RandomForestRegressor()
        regressor.fit(self.DE_dfX, self.DE_dfY)
        DEX_train, DEX_test, DEy_train, DEy_test = train_test_split(self.DE_dfX, self.DE_dfY, test_size=0.3,random_state=42)
        predictions = regressor.predict(DEX_test)
        print(predictions)
        sns.displot(DEy_test - predictions)
        plt.show()
        print(regressor.best_params_)
        print(regressor.best_score_)
        rmse = np.sqrt(mean_squared_error(DEy_test, predictions))
        print("rmse",rmse)
        print("R2 score : %.2f" % r2_score(DEy_test, predictions))
        res= stats.spearmanr(DEy_test, predictions)
        print(res)
    def RandomForestregressionFR(self):
        regressor = RandomForestRegressor()
        regressor.fit(self.FR_dfX, self.FR_dfY)
        FRX_train, FRX_test, FRy_train, FRy_test = train_test_split(self.FR_dfX, self.FR_dfY, test_size=0.3,random_state=42)
        predictions = regressor.predict(FRX_test)
        print(predictions)
        sns.displot(FRy_test - predictions)
        plt.show()
        print(regressor.best_params_)
        print(regressor.best_score_)
        rmse = np.sqrt(mean_squared_error(FRy_test, predictions))
        print("rmse",rmse)
        print("R2 score : %.2f" % r2_score(FRy_test, predictions))
        res= stats.spearmanr(FRy_test, predictions)
        print(res)



