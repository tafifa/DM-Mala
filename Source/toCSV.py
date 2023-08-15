import pandas as pd

import firstClassification as fst
import secondClassification as sec
import distanceComparison as dcom

def createCSV_Infected():
	# PARASITIZED
	dfO = fst.getDataFrame('../Data/Infeksi/Parasitized/*')

  # UNINFECTED
	dfM = fst.getDataFrame('../Data/Infeksi/Uninfected/*')
	# print(dfO, dfM)

	df_concat = pd.concat([dfO, dfM], ignore_index=True)
	df_concat.to_csv('csv/infectedClassification.csv')

def createCSV_Stadium():
	# SEHAT
	df_Sehat = sec.getDataFrame('../Data/Stadium/Sehat/*')

	# STADIUM 1
	df_Tropozoit = sec.getDataFrame('../Data/Stadium/Tropozoit/*')

	# STADIUM 2
	df_Gametosit = sec.getDataFrame('../Data/Stadium/Gametosit/*')

	# STADIUM 3
	df_Sizon = sec.getDataFrame('../Data/Stadium/Sizon/*')

	df_concat = pd.concat([df_Sehat, df_Tropozoit, df_Gametosit, df_Sizon], ignore_index=True)
	df_concat.to_csv('csv/stadiumClassification.csv')

def createCSV_result(dir, metricOpt):
	df_result = dcom.getDataFrame(dir, metricOpt)

	df_result.to_csv('csv/result.csv')