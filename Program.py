# imports statements
from Classes.ModelProcessor import ModelProcessor
import sys
from Enums.EFile import EFile
from Classes.DataProcessor import DataProcessor
from Classes.DataVisualizer import DataVisualizer

# Visualize data
# dataVisualizer = DataVisualizer(processedDf)
# dataVisualizer.GenerateScatterPlots()
# dataVisualizer.GenerateHistograms()
# dataVisualizer.GenerateMultiPlots()

# Process csv
dataProcessor = DataProcessor(EFile.PLAYER)

modelProcessor = ModelProcessor(dataProcessor.GetDataFrames())

print(modelProcessor.GetLogisticCoefficients())
# processedDf = dataProcessor.GetDataFrame()
