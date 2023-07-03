import pandas as pd

from dataProcessor import getDataFrame

def createDatabase():
	# PARASITIZED
	dfO = getDataFrame('../Data/cell_images/Parasitized/*')

  # UNINFECTED
	dfM = getDataFrame('../Data/cell_images/Uninfected/*')
	# print(dfO, dfM)

	df_concat = pd.concat([dfO, dfM], ignore_index=True)
	df_concat.to_csv('csv/KBase.csv')

def createDatabase2():
	# GAMETOSIT
	dfG = getDataFrame('../Data/Dataset/Gametosit/*')

  # SEHAT
	dfS = getDataFrame('../Data/Dataset/Sehat/*')

	# SIZON
	dfSz = getDataFrame('../Data/Dataset/Sizon/*')

	# TROPOZOIT
	dfT = getDataFrame('../Data/Dataset/Tropozoit/*')

	# print(dfG, dfS, dfSz, dfT)

	df_concat = pd.concat([dfG, dfS, dfSz, dfT], ignore_index=True)
	df_concat.to_csv('csv/KBase2.csv')