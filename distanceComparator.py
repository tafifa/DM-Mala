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

	x = np.array(data.iloc[:, 0:6])
	y = np.array(data.iloc[:, 7]).ravel()

	# print(x,y)

	knn = KNeighborsClassifier(metric=metricOpt, n_neighbors=1)
	knn.fit(x, y)

	# print(knn)

	path = glob.glob(pathDir)
	i = 0

	## test time
	start_time = time.time()
    
	for item in path:
		if i == 4799:
				break
		# print(item)
		# i += 1

		glcmData = getData(item)
		inputTest = np.array(glcmData).reshape(1, -1)
		# print(inputTest)
		result = knn.predict(inputTest).reshape(1, -1)
		filename = Path(item).stem

		# print('hasil ' , result, filename)
		print(result[0])

		# if result == 'Uninfected':
		# 	i += 1
		# 	print('hasil ' , result, filename)
		# 	shutil.copy(item, '../Data/cell_images/dummy/Uninfected/')
	
	# print("\nTime elapsed for distance comparison: {:.4f}s".format(time.time() - start_time))
	# print("Total gambar yang  dilooping didapatkan sebanyak", i, "gambar")

def distanceComparison2(dir):
	data = pd.read_csv('csv/KBase2.csv')

	x = np.array(data.iloc[:, 0:10])
	y = np.array(data.iloc[:, 11]).ravel()

	path = glob.glob(dir)

	## test time
	i = 0
	start_time = time.time()

	for item in path:
		minimal = 10e14
		if i == 4799:
				break
		# print(item)
		i += 1

		glcmData = getData(item)
		inputTest = np.array(glcmData)[0]
		c = ''
		# print(inputTest)
		for i in range (len(x)):
			accuracy = dist.canberra(inputTest, x[i])
			# print(accuracy)
			minimal = min(accuracy, minimal)
			if minimal == accuracy:
				c = y[i]
				
		filename = Path(item).stem
		print('hasil ' , c, filename)

		# filename = Path(item).stem

	# 	print('hasil ' , result, filename)

	# print("\nTime elapsed for distance comparison: {:.4f}s".format(time.time() - start_time))
	# print("Total gambar yang  dilooping didapatkan sebanyak", i, "gambar")
