import glob
import time
import cv2
import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis, entropy

from segmentation import segmentation

def gethistogram(img):

	# Calculate entropy
	histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
	probabilities = histogram / np.sum(histogram)
	entropyVal = entropy(probabilities)[0]

	# Calculate mean
	meanVal = np.mean(img)

	# Calulcate variance
	varianceVal = np.mean(img)

	# Flatten the img array for skewness and kurtosis calculation
	flattened_img = img.flatten()

	skewnessVal = skew(flattened_img)

	kurtosisVal = kurtosis(flattened_img)

	if np.all(flattened_img == 0):
		skewnessVal , kurtosisVal = 0, 0

	# Calculate standar deviation
	std_devVal = np.std(img)

	# feature = [entropyVal, meanVal, varianceVal, std_devVal, skewnessVal, kurtosisVal]
	feature = [skewnessVal, kurtosisVal]
	
	# print(feature)
	return feature

def getglcm(img):
	# properties = [ 'contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation' ]
	# properties = [ 'contrast', 'correlation', 'energy', 'homogeneity' ]
	properties = [ 'homogeneity', 'correlation' ]

	glcm = graycomatrix(img,
		     							distances=[1],
											angles = [0],
											levels = 256,
											symmetric = True,
											normed = True)
	
	feature = []
	glcm_props = [propery for name in properties for propery in graycoprops(glcm, name)[0]]
	for item in glcm_props:
		feature.append(item)
	
	# print(feature)
	return feature

def getData(dir):
	path = glob.glob(dir)
	data = []

	for item in path:
		filename = Path(item).stem

		img  = cv2.imread(item, cv2.IMREAD_COLOR)
		img2, _, _, roundness = segmentation(img)
		gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

		glcm = getglcm(gray)
		hist = gethistogram(gray)

		temp = np.concatenate([[roundness], glcm, hist])

		list = temp.tolist()
		list.insert(0, filename)

		# print(list)
		
		data.append(list)

	# print(data)
	return data

def getDataFrame(path):	

	# glcm_feature = [ 'filename', 'contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation' ]
	# histogram_feature = ['filename', 'entropy', 'mean', 'variance', 'std_dev', 'skewness', 'kurtosis']
	# texture_feature = [ 'filename', 'contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation', 'entropy', 'mean', 'variance', 'std_dev', 'skewness', 'kurtosis' ]
	# texture_feature = [ 'filename', 'contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation', 'entropy', 'mean', 'variance', 'std_dev', 'skewness', 'kurtosis' ]
	txt_feature = [ 'filename', 'roundness', 'homogeneity', 'correlation', 'skewness', 'kurtosis']

	glcm_all_agls = getData(path)

	glcm_df = pd.DataFrame(glcm_all_agls, columns=txt_feature)

	glcm_df = glcm_df.assign(label=os.path.basename(os.path.dirname(path)))
	# print(glcm_df)

	return glcm_df

if __name__ == '__main__':
	dir = '../Data/Fase/test/*'

	print(getDataFrame(dir))