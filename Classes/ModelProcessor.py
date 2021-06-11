
from typing import List
import pandas as pd
from sklearn.linear_model import LinearRegression


class ModelProcessor:
    def __init__(self, dataframes: List) -> None:
        self.__x_train__, self.__x_test__, self.__y_train__, self.__y_test__ = dataframes
        self.__MakeNpArraysDataFrames__()
        self.__ExecuteLinearRegression__()

    def __MakeNpArraysDataFrames__(self):
        self.__x_train__ = pd.DataFrame(self.__x_train__)
        self.__x_test__ = pd.DataFrame(self.__x_test__)
        self.__y_train__ = pd.DataFrame(self.__y_train__)
        self.__y_test__ = pd.DataFrame(self.__y_test__)

    def __ExecuteLinearRegression__(self):
        self.__regressor__ = LinearRegression()
        self.__regressor__.fit(self.__x_train__, self.__y_train__)

    def PredictLinearRegression(self):
        y_pred = pd.DataFrame(self.__regressor__.predict(self.__x_train__))

        result = pd.concat([self.__y_test__.head(), y_pred.head()],
                           ignore_index=True, sort=False, axis=1)
        print(result)
