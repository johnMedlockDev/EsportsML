from typing import List
import pandas as pd
from pandas import DataFrame

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Enums.EFile import EFile


class DataProcessor:

    def __init__(self, file: EFile) -> None:
        self.__fileContext__ = file.value
        self.__ReadCSV__()
        self.__ReplaceCategoricalDataWithDummyData__()
        self.__SelectFeatures__()
        self.__SplitData__()
        self.__NormalizeDataFrames__()
        self.__MakeNpArraysDataFrames__()

    def __ReadCSV__(self) -> None:

        self.__df__ = pd.read_csv(f"Data\\{self.__fileContext__}.csv")

    def __SplitData__(self) -> None:

        x_data = self.__df__[self.__dfColumnNames__]
        y_data = np.array(self.__df__['result']).reshape(-1, 1)

        self.__x_train__, self.__x_test__, self.__y_train__, self.__y_test__ = train_test_split(
            x_data, y_data, test_size=0.3, random_state=42)

    def __SelectFeatures__(self) -> None:

        if self.__fileContext__ == "Player":
            self.__dfColumnNames__ = ["kills", "deaths",
                                      "assists",
                                      ]

        if self.__fileContext__ == "Team":
            self.__dfColumnNames__ = ["dragons", "kills",	"deaths", "assists",
                                      ]

    def __ReplaceCategoricalDataWithDummyData__(self) -> None:
        self.__df__ = pd.get_dummies(self.__df__,
                                     columns=['position', 'side'])

    def __NormalizeDataFrames__(self) -> None:
        scaler = StandardScaler()
        self.__x_train__ = scaler.fit_transform(self.__x_train__)

        self.__x_test__ = scaler.fit_transform(self.__x_test__)

    def DisplayColumns(self) -> None:
        for column in self.__df__.columns:
            print(column)

    def __MakeNpArraysDataFrames__(self) -> None:

        self.__x_train__ = pd.DataFrame(self.__x_train__)
        self.__x_train__.columns = self.__dfColumnNames__

        self.__x_test__ = pd.DataFrame(self.__x_test__)
        self.__x_test__.columns = self.__dfColumnNames__

        self.__y_train__ = pd.DataFrame(self.__y_train__)
        self.__x_test__.columns = self.__dfColumnNames__

        self.__y_test__ = pd.DataFrame(self.__y_test__)
        self.__x_test__.columns = self.__dfColumnNames__

    def GetDataFrames(self) -> List:
        return [self.__x_train__, self.__x_test__, self.__y_train__, self.__y_test__]
