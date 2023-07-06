import pandas as pd
import numpy as np
import glob
import time
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
import scipy.spatial.distance as dist

from dataProcessor import getData

def distanceComparison(pathDir, metricOpt):
	data = pd.read_csv('csv/KBase2.csv')

	x = np.array(data.iloc[:, 0:10])
	y = np.array(data.iloc[:, 11]).ravel()

	knn = KNeighborsClassifier(metric=metricOpt, n_neighbors=1)
	knn.fit(x, y)

	path = glob.glob(pathDir)

	## test time
	start_time = time.time()
    
	for item in path:
		glcmData = getData(item)
		inputTest = np.array(glcmData).reshape(1, -1)
		result = knn.predict(inputTest).reshape(1, -1)
		filename = Path(item).stem
		c = result[0][0]

		print('hasil' , c, filename)
	
	# print("\nTime elapsed for distance comparison: {:.4f}s".format(time.time() - start_time))
	# print("Total gambar yang  dilooping didapatkan sebanyak", i, "gambar")

def distanceComparison2(dir):
	data = pd.read_csv('csv/KBase2.csv')

	x = np.array(data.iloc[:, 0:10])
	y = np.array(data.iloc[:, 11]).ravel()

	path = glob.glob(dir)

	## test time
	start_time = time.time()

	for item in path:
		minimal = 10e14

		glcmData = getData(item)
		inputTest = np.array(glcmData)[0]
		c = ''
		for i in range (len(x)):
			accuracy = dist.canberra(inputTest, x[i])
			minimal = min(accuracy, minimal)
			if minimal == accuracy:
				c = y[i]
				
		filename = Path(item).stem
		print('hasil' , c, filename)

	# print("\nTime elapsed for distance comparison: {:.4f}s".format(time.time() - start_time))
	# print("Total gambar yang  dilooping didapatkan sebanyak", i, "gambar")
