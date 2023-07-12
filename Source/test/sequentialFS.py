import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def seqfs():
	
	data = pd.read_csv('../csv/stadiumClassification2.csv')
	texture_feature = [ 'contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation', 'entropy', 'mean', 'variance', 'std_dev', 'skewness', 'kurtosis' ]
	txt_feature = ['homogeneity', 'correlation', 'skewness', 'kurtosis']
	
	x = np.array(data[texture_feature])
	scaler = StandardScaler()
	x_scaled = scaler.fit_transform(x)

	y = np.array(data[['label']]).ravel()

	# Initialize model and SequentialFeatureSelector
	model = LogisticRegression(solver='liblinear', max_iter=1000)
	sfs = SequentialFeatureSelector(model, n_features_to_select=4, direction='backward')

	# Perform sequential forward selection
	sfs.fit(x, y)

	# Get selected feature indices
	selected_feature_indices = sfs.get_support(indices=True)

	slct = selected_feature_indices.tolist()
	print(slct)

	for i in range (len(slct)):
		print(texture_feature[slct[i]])

if __name__ == '__main__':
	seqfs()