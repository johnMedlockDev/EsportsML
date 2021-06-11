
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt


class DataVisualizer:

    def __init__(self, dataframe: DataFrame) -> None:
        self.__df__ = dataframe
        self.columns = [ele for ele in self.__df__.columns if ele not in [
            "result", "player", "team", "position"]]

    def GenerateCorrelationMatrix(self):
        corrMatrix = self.__df__.corr()
        sns.heatmap(corrMatrix, annot=True)
        plt.show()

    def GenerateHistograms(self):

        for column in self.columns:
            sns.histplot(data=self.__df__, y=column)
            plt.show()

    def GenerateScatterPlots(self, value: str = "earnedgold"):

        for column in self.columns:
            sns.scatterplot(data=self.__df__, x=column, y=value)
            plt.show()

    def GenerateLinePlots(self, value: str = "result"):

        for column in self.columns:
            sns.lineplot(data=self.__df__, x=column, y=value)
            plt.show()

    def GenerateMultiPlots(self):
        g = sns.PairGrid(self.__df__, hue='position')
        g.map_diag(sns.histplot)
        g.map_offdiag(sns.scatterplot)
        g.add_legend()
        plt.show()
