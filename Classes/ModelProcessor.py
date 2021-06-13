
from typing import List
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


class ModelProcessor:
    def __init__(self, dataframes: List) -> None:
        self.__x_train__, self.__x_test__, self.__y_train__, self.__y_test__ = dataframes
        self.__ExecuteLinearRegression__()
        self.__ExecuteLogisticRegression__()

    def __ExecuteLinearRegression__(self) -> None:
        self.__linearRegressor__ = LinearRegression()
        self.__linearRegressor__.fit(self.__x_train__, self.__y_train__)

    def __ExecuteLogisticRegression__(self) -> None:
        self.__logisticRegression__ = LogisticRegression(max_iter=10000)
        self.__logisticRegression__.fit(
            self.__x_train__, self.__y_train__.values.ravel())

    def GetPredictionLinearRegression(self) -> pd.DataFrame:
        y_pred = pd.DataFrame(
            self.__linearRegressor__.predict(self.__x_train__))

        df = pd.concat([self.__y_test__.head(), y_pred.head()],
                       ignore_index=True, sort=False, axis=1)
        return df

    def GetPredictionLogisticRegression(self) -> pd.DataFrame:
        predictions = self.__logisticRegression__.predict(self.__x_test__)

        df = pd.concat([self.__x_test__, self.__y_test__,
                        pd.DataFrame(predictions)], sort=False, axis=1)

        df.columns = self.__AddColumnNames__()

        return df

    def __AddColumnNames__(self):
        columns = self.__x_test__.columns.tolist()
        columns.append("result")
        columns.append("prediction")
        return columns

    def GetLogisticAccuracyScore(self) -> float:
        score = self.__logisticRegression__.score(
            self.__x_test__, self.__y_test__)
        return score

    def GetLogisticCoefficients(self) -> pd.DataFrame:
        log_odds = self.__logisticRegression__.coef_[0]
        df = pd.DataFrame(log_odds,
                          self.__x_test__.columns,
                          columns=['coef']).sort_values(by='coef', ascending=False)

        return df
