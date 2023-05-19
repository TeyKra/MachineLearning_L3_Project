import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
import seaborn as sns
import sklearn.cluster as skc
import sklearn.preprocessing as skp
import csv
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from Modeledonnee import DataLoad

class Regression():
    def __init__(self,data):

        FRX_train, FRX_test, FRy_train, FRy_test = train_test_split(data.FR_dfX, data.FR_dfY, test_size=0.3, random_state=42)
        DEX_train, DEX_test, DEy_train, DEy_test = train_test_split(data.DE_dfX, data.DE_dfY, test_size=0.3, random_state=42)
        model = LinearRegression()

        model.fit(FRX_train, FRy_train)
        model.fit(DEX_train, DEy_train)
        FRy_pred = model.predict(FRX_test)
        DEy_pred = model.predict(DEX_test)
        print(FRy_pred,DEy_pred, model.coef_, model.intercept_, mean_squared_error(FRy_test, FRy_pred), mean_squared_error(DEy_test,DEy_pred), r2_score(FRy_test, FRy_pred), r2_score(DEy_test,DEy_pred))
    print("--------------------------------------------------------------------------------------------------------------")

    def showCorrelation(self,data):
        """
        Affiche les correlations entre les variables
        """
        sns.heatmap(data.FR_dfX.corr())
        plt.show()
        sns.heatmap(data.DE_dfX.corr())
        plt.show()
