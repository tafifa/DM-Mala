import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis, entropy
import glob
import time
import os
import pandas as pd
from segmentationImg import segmentation

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
		img  = cv2.imread(item, cv2.IMREAD_COLOR)
		
		imged = segmentation(img)

		gray = cv2.cvtColor(imged, cv2.COLOR_RGB2GRAY)

		# show(gray)
		
		imgs.append(gray)

	glcm_all_agls = []
	for img in imgs: 
		glcm_all_agls.append(getglcm2(img))

	# print("\nTime elapsed for getting data: {:.4f}s".format(time.time() - start_time))
	# print("Total gambar yang sudah di looping sebanyak", i, "gambar")
	
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