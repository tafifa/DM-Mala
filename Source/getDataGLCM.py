from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
import glob
import os
import shutil
import pandas as pd
from pathlib import Path
import time
import scipy.spatial.distance as dist
from scipy.stats import skew, kurtosis, entropy

properties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
properties2 = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation','Std Dev', 'Entropy', 'Skewness', 'Kurtosis']

def show(img):
	cv2.imshow("test img", img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

def getglcm(img):
	# agls = [0, np.pi/4, np.pi/2, 3*np.pi/4]
	agls = [np.pi/4]
	glcm = graycomatrix(img,
		     							distances=[1],
											angles = agls,
											levels = 256,
											symmetric = True,
											normed = True)
	
	# print(glcm)
	
	feature = []
	glcm_props = [propery for name in properties for propery in graycoprops(glcm, name)[0]]
	for item in glcm_props:
		feature.append(item)
	
	return feature

def getglcm2(img):
	# agls = [0, np.pi/4, np.pi/2, 3*np.pi/4]
	agls = [np.pi/4]
	glcm = graycomatrix(img,
		     							distances=[1],
											angles = agls,
											levels = 256,
											symmetric = True,
											normed = True)
	
	# print(glcm)
	
	feature = []
	glcm_props = [propery for name in properties for propery in graycoprops(glcm, name)[0]]
	for item in glcm_props:
		feature.append(item)
	
	glco = []
	std_dev = np.std(img)

	# Calculate entropy
	histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
	probabilities = histogram / np.sum(histogram)
	entropy_value = entropy(probabilities)

	# Flatten the img array for skewness and kurtosis calculation
	flattened_img = img.flatten()

	# Calculate skewness
	skewness = skew(flattened_img)

	# Calculate kurtosis
	kurtosis_value = kurtosis(flattened_img)

	glco = [std_dev, entropy_value[0], skewness, kurtosis_value]

	feature += glco
	
	return feature

def getData(pathDir):
	path = glob.glob(pathDir)
	imgs = []
	i = 0

	## test time
	start_time = time.time()
    
	for item in path:
		if i == 4799:
				break
		# print(item)
		i += 1
		img  = cv2.imread(item, cv2.IMREAD_COLOR)
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		# THRESHOLDING
		_, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

		# Set the black regions to white in the original image
		img[mask == 0] = [255, 255, 255]  # Set black pixels to white (BGR value)
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		# show(gray)
		
		imgs.append(gray)

	glcm_all_agls = []
	for img in imgs: 
		glcm_all_agls.append(getglcm2(img))
	
	# print(glcm_all_agls)

	# print("\nTime elapsed for getting data: {:.4f}s".format(time.time() - start_time))
	# print("Total gambar yang sudah di looping sebanyak", i, "gambar")

	# print(glcm_all_agls)
	
	return glcm_all_agls

def getDataFrame(path):	
	columns = []
	# angles = ['0', '45', '90','135']
	# for name in properties :
	# 		for ang in angles:
	# 				columns.append(name + "_" + ang)

	for name in properties2 :
		columns.append(name)

	glcm_all_agls = getData(path)
	
	glcm_df = pd.DataFrame(glcm_all_agls, columns=columns)
	glcm_df = glcm_df.assign(label=os.path.basename(os.path.dirname(path)))

	return glcm_df

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

if __name__ == '__main__':
	# createDatabase()
	dir = '../Data/test/*'
	# print(getData(dir))
	# createDatabase2()
	# print(getData(dir))
	# distanceComparison(dir, 'canberra')
	distanceComparison2(dir)
	# print("Hello")
	
'''

the problem is glcm program calculate the background too, and it affect in the knowledge base
it has been tested with white and black background and return different value

new objective 19/6/2023 13.59
- create database with 2400 data images or more
- then find the true positive and false negative parasitized and uninfected folder
- then save true positive and false negative to dummy folder in each folder
- create database again with same amount of images following the smallest total images
- follow step 2 and 3 respectively, again and again until distance comparison make a good predict
- if it fail, you can try from step 1 with bigger amount data images
- if it fail again, you must consider with thresolding, feature selection, angles selection, and maybe coding review from first
- expectation finishing it will be 1-2 days

new objective 26/6/2023 23.42
- objective before didnt work because that segmentation and thresolding didnt do well
- we need new segmentation and thresolding that can handle classification better

'''