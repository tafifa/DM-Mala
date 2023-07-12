import glob
import time
import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import svm
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

from segmentation import segmentation
from secondClassification import getglcm, gethistogram

def stadium_classificationKNN(dir, metricOpt):
	path = glob.glob(dir)

	# data = pd.read_csv('csv/stadiumClassification.csv')
	# x = np.array(data[[ 'contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation' ]])
	
	data = pd.read_csv('csv/stadiumClassification.csv')
	# texture_feature = [ 'contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation', 'entropy', 'mean', 'variance', 'std_dev', 'skewness', 'kurtosis' ]
	txt_feature = [ 'roundness', 'homogeneity', 'correlation', 'skewness', 'kurtosis']

	x = np.array(data[txt_feature])
	y = np.array(data[['label']]).ravel()

	knn = KNeighborsClassifier(metric=metricOpt, n_neighbors=1)
	knn.fit(x, y)

	result = []
	for item in path:
		filename = Path(item).stem

		img  = cv2.imread(item, cv2.IMREAD_COLOR)
		img2, _, _, roundness = segmentation(img)
		gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

		glcm = getglcm(gray)
		hist = gethistogram(gray)
		
		temp = np.concatenate([[roundness], glcm, hist])

		inputTest = np.array(temp).reshape(1,-1)
		# print(inputTest)
		prediction = knn.predict(inputTest).reshape(1, -1)

		list = temp.tolist()
		list.insert(0, filename)
		list.append(prediction[0][0])

		# print(list)
		result.append(list)

	return result

def stadium_classificationSVM(dir, metricOpt):
	path = glob.glob(dir)

	data = pd.read_csv('csv/stadiumClassification.csv')
	# texture_feature = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation', 'entropy', 'mean', 'variance', 'std_dev', 'skewness', 'kurtosis']
	txt_feature = [ 'roundness', 'homogeneity', 'correlation', 'skewness', 'kurtosis']

	x = np.array(data[txt_feature])
	y = np.array(data[['label']]).ravel()

	svm_model = svm.SVC(kernel='linear')
	svm_model.fit(x, y)

	result = []
	for item in path:
		filename = Path(item).stem

		img  = cv2.imread(item, cv2.IMREAD_COLOR)
		img2, _, _, roundness = segmentation(img)
		gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

		glcm = getglcm(gray)
		hist = gethistogram(gray)
		
		temp = np.concatenate([[roundness], glcm, hist])

		inputTest = np.array(temp).reshape(1, -1)
		prediction = svm_model.predict(inputTest).reshape(1, -1)

		data_list = temp.tolist()
		data_list.insert(0, filename)
		data_list.append(prediction[0][0])

		result.append(data_list)

	return result


def getDataFrame(path, metricOpt):
	# glcm_feature = [ 'filename', 'contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation', 'prediction' ]
	# histogram_feature = ['filename', 'entropy', 'mean', 'variance', 'std_dev', 'skewness', 'kurtosis', 'prediction']
	# texture_feature = [ 'filename', 'contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation', 'entropy', 'mean', 'variance', 'std_dev', 'skewness', 'kurtosis', 'prediction' ]
	txt_feature = [ 'filename', 'roundness', 'homogeneity', 'correlation', 'skewness', 'kurtosis', 'prediction']

	glcm_all_agls = stadium_classificationKNN(path, metricOpt)

	glcm_df = pd.DataFrame(glcm_all_agls, columns=txt_feature)
	# print(glcm_df)

	return glcm_df