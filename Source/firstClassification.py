### THIS CLASSIFICATION USING OBJECT COUNTING TO CLASSIFYING INFECTED AND UNINFECTED CELL IMAGES

import pandas as pd
import os
import glob
import cv2
from pathlib import Path

from segmentation import segmentation

def infected_classification(path):
	dirPath = glob.glob(path)

	res = []

	for item in dirPath:
		filename = Path(item).stem

		img  = cv2.imread(item, cv2.IMREAD_COLOR)

		_, count, _, _ = segmentation(img)

		if count > 0:
			res.append([filename, count, 'Parasitized'])

		else:
			res.append([filename, count, 'Uninfected'])
	
	return res

def getDataFrame(path):

	cls = infected_classification(path)

	label = ['filename', 'count', 'result']


	df = pd.DataFrame(cls, columns=label)

	df = df.assign(label=os.path.basename(os.path.dirname(path)))

	return df