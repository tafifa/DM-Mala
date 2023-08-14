import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read(image):
	img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	imgplot = plt.imshow(img)
	plt.show()

def masking(img, l, u):
	hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

	lower_threshold = np.array(l, dtype=np.uint8)
	upper_threshold = np.array(u, dtype=np.uint8)

	mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)
	segmented_image = cv2.bitwise_and(img, img, mask=mask)
	
	return segmented_image

def rectangle(mask, imged):
	img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(img, 10, 255, 0)
	contours, _ = cv2.findContours(thresh, 1, 2)

	xmin, xmax, ymin, ymax = [], [], [], []

	if len(contours) < 1:
		return mask, 0, 0, 0
	
	count = 0
	roundnessVal = 0
	for i, cnt in enumerate(contours):
		area = cv2.contourArea(cnt)
		perimeter = cv2.arcLength(cnt, True)

		M = cv2.moments(cnt)
		if M['m00'] != 0.0 and area > 80:
			count += 1
			roundness = (4 * np.pi * area) / (perimeter ** 2)
			roundnessVal += roundness
			
			### UNCOMMENT INI UNTUK MENDAPATKAN BATAS LUASAN PARASIT
			# img1 = cv2.drawContours(imged, [cnt], -1, (0, 255, 255), 1)

			x, y, w, h = cv2.boundingRect(cnt)

			xmin.append(x)
			xmax.append(x+w)
			ymin.append(y)
			ymax.append(y+h)

	if count > 0:
		### UNTUK MENDAPATKAN GAMBAR PARASIT DENGAN UKURAN YANG DI CROP
		crop = mask[min(ymin):max(ymax), min(xmin):max(xmax)] 

		### UNTUK MENDAPATKAN GAMBAR PARASIT DENGAN UKURAN ASLI
		# crop = mask

		return crop, count, area, roundnessVal
	
	else:
		return mask, 0, 0, 0

def segmentation(img):

	l =  [57, 147, 93]	# BATAS LOWER RANGE ATAU GELAP
	u = [173, 194, 128]	# BATAS UPPER RANGE ATAU TERANG

	mask = masking(img, l, u)

	imgcrop, count, area, roundness = rectangle(mask, img)

	return imgcrop, count, area, roundness

if __name__ == '__main__':

	dirpath = '../Data/Stadium/test/*'
	path = glob.glob(dirpath)

	i = 0
	for item in path:
		i += 1
		
		img  = cv2.imread(item, cv2.IMREAD_COLOR)

		imgcrop, count, area, roundness = segmentation(img)

		### UNTUK MELIHAT HASIL CROP
		read(imgcrop)

		## UNTUK MENGETAHUI JUMLAH PARASIT
		print(count)

'''

'''