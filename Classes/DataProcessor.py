from typing import List
import pandas as pd
from pandas import DataFrame

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from Enums.EFile import EFile


class DataProcessor:

    def __init__(self, file: EFile) -> None:
        self.__fileContext__ = file.value
        self.__ReadCSV__()
        self.__DropColumns__()
        self.__ReplaceCatsWithDummies__()
        self.__SplitData__()
        self.__NormalizeDataFrames__()

    def __ReadCSV__(self) -> None:

        self.__df__ = pd.read_csv(f"Data\\{self.__fileContext__}.csv")

    def __DropColumns__(self) -> None:

        columnsToDrop = self.__GetColumnsToDrop__()
        self.__df__ = self.__df__.drop(columns=columnsToDrop)

    def __GetColumnsToDrop__(self) -> List:

        if self.__fileContext__ == "Player":
            return ["datacompleteness",
                    "url",
                    "league",
                    "split",
                    "playoffs",
                    "year",
                    "patch",
                    "side",
                    "playerid",
                    "ban1",
                    "ban2",
                    "ban3",
                    "ban4",
                    "ban5",
                    "firstdragon",
                    "dragons",
                    "opp_dragons",
                    "elementaldrakes",
                    "opp_elementaldrakes",
                    "infernals",
                    "mountains",
                    "clouds",
                    "oceans",
                    "dragons (type unknown)",
                    "elders",
                    "opp_elders",
                    "firstherald",
                    "heralds",
                    "opp_heralds",
                    "firstbaron",
                    "barons",
                    "opp_barons",
                    "firsttower",
                    "towers",
                    "opp_towers",
                    "firstmidtower",
                    "firsttothreetowers",
                    "inhibitors",
                    "opp_inhibitors",
                    "gspd",
                    "teamkills",
                    "teamdeaths",
                    "goldat15",
                    "xpat15",
                    "csat15",
                    "killsat15",
                    "assistsat15",
                    "deathsat15",
                    "dpm",
                    "damageshare",
                    "damagetakenperminute",
                    "damagemitigatedperminute",
                    "wpm",
                    "wardskilled",
                    "wcpm",
                    "controlwardsbought",
                    "visionscore",
                    "vspm",
                    "totalgold",
                    "earned gpm",
                    "earnedgoldshare",
                    "goldspent",
                    "minionkills",
                    "monsterkills",
                    "monsterkillsownjungle",
                    "monsterkillsenemyjungle",
                    "cspm",
                    "golddiffat10",
                    "xpdiffat10",
                    "csdiffat10",
                    "golddiffat15",
                    "xpdiffat15",
                    "csdiffat15",
                    "team kpm",
                    "ckpm",
                    "opp_goldat15",
                    "opp_xpat15",
                    "opp_csat15",
                    "opp_killsat15",
                    "opp_assistsat15",
                    "opp_deathsat15",
                    "opp_killsat10",
                    "opp_assistsat10",
                    "opp_deathsat10",
                    "opp_goldat10",
                    "opp_xpat10",
                    "opp_csat10",
                    "date",
                    "champion",
                    "goldat10",
                    "xpat10",
                    "csat10",
                    "killsat10",
                    "assistsat10",
                    "deathsat10",
                    "game",
                    "doublekills",
                    "triplekills",
                    "quadrakills",
                    "pentakills",
                    "firstblood",
                    "firstbloodkill",
                    "firstbloodassist",
                    "firstbloodvictim",
                    "gamelength", "gameid", "player", "team"]

        return ["gameid",
                "datacompleteness",
                "url",
                "league",
                "year",
                "split",
                "playoffs",
                "date",
                "patch",
                "playerid",
                "side",
                "position",
                "player",
                "champion",
                "ban1",
                "ban2",
                "ban3",
                "ban4",
                "ban5",
                "firstbloodkill",
                "firstbloodassist",
                "firstbloodvictim",
                "dragons (type unknown)",
                "damageshare",
                "earnedgoldshare",
                "total cs",
                "dpm",
                "damageshare",
                "damagetakenperminute",
                "damagemitigatedperminute",
                "wpm",
                "wardskilled",
                "wcpm",
                "controlwardsbought",
                "visionscore",
                "vspm",
                "totalgold",
                "earned gpm",
                "earnedgoldshare",
                "goldspent",
                "minionkills",
                "monsterkills",
                "monsterkillsownjungle",
                "monsterkillsenemyjungle",
                "cspm",
                "golddiffat10",
                "xpdiffat10",
                "csdiffat10",
                "golddiffat15",
                "xpdiffat15",
                "csdiffat15",
                "team kpm",
                "ckpm",
                "opp_goldat15",
                "opp_xpat15",
                "opp_csat15",
                "opp_killsat15",
                "opp_assistsat15",
                "opp_deathsat15",
                "opp_killsat10",
                "opp_assistsat10",
                "opp_deathsat10",
                "opp_goldat10",
                "opp_xpat10",
                "opp_csat10",
                "date",
                "champion",
                "goldat10",
                "xpat10",
                "csat10",
                "killsat10",
                "assistsat10",
                "deathsat10",
                "gamelength",
                "game", "player", "team"]

    def __SplitData__(self) -> None:

        x_data = self.__df__[["position_bot", "position_jng", "position_mid", "position_sup", "position_top", "kills", "deaths",
                              "assists", "damagetochampions", "wardsplaced", "earnedgold", "total cs"]]

        y_data = np.array(self.__df__['result']).reshape(-1, 1)

        self.__x_train__, self.__x_test__, self.__y_train__, self.__y_test__ = train_test_split(
            x_data, y_data, test_size=0.3, random_state=42)

    def __ReplaceCatsWithDummies__(self) -> None:

        self.__df__ = pd.get_dummies(self.__df__,
                                     columns=['position'])

    def __NormalizeDataFrames__(self) -> None:
        scaler = StandardScaler()
        self.__x_train__ = scaler.fit_transform(self.__x_train__)
        self.__x_test__ = scaler.fit_transform(self.__x_test__)

    def DisplayColumns(self) -> None:
        for column in self.__df__.columns:
            print(column)

    def GetXTrainDataFrame(self) -> DataFrame:
        return self.__x_train__

    def GetXTestDataFrame(self) -> DataFrame:
        return self.__x_test__

    def GetYTrainDataFrame(self) -> DataFrame:
        return self.__y_train__

    def GetYTestDataFrame(self) -> DataFrame:
        return self.__y_test__

    def GetDataFrames(self):
        return [self.__x_train__, self.__x_test__, self.__y_train__, self.__y_test__]
