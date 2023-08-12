import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read(dir, img):
	if dir != '':
		img = cv2.imread(dir)
		im_rgb = cv2.cvtColor(dir, cv2.COLOR_BGR2HSV)
	else:
		im_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
		# cv2.imwrite(f'lab{i}.png', im_rgb)
	imgplot = plt.imshow(im_rgb)
	plt.show()

def show(img):
	imgplot = plt.imshow(img)
	plt.show()

def masking(img, l, u):

	# Convert the image to the HSV color space
	hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

	# Define the lower and upper threshold values for the color range
	lower_threshold = np.array(l, dtype=np.uint8)  # Lower threshold for the color range (in HSV)
	upper_threshold = np.array(u, dtype=np.uint8)  # Upper threshold for the color range (in HSV)

	# Create a binary mask by applying the color threshold
	mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)

	# Apply the mask to the original image
	segmented_image = cv2.bitwise_and(img, img, mask=mask)

	# show(segmented_image)
	# show(hsv_image)
	
	return segmented_image

def rectangle(mask, imged):

	real = imged
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
				
			# img1 = cv2.drawContours(imged, [cnt], -1, (0, 255, 255), 1)
			x, y, w, h = cv2.boundingRect(cnt)

			xmin.append(x)
			xmax.append(x+w)
			ymin.append(y)
			ymax.append(y+h)

	if count > 0:
		crop = real[min(ymin):max(ymax), min(xmin):max(xmax)]
		# crop = mask

		# print('Jumlah kontur', count)

		return crop, count, area, roundnessVal
	
	else:
		return mask, 0, 0, 0

def segmentation(img):

	# BATAS LOWER RANGE ATAU GELAP
	l =  [57, 147, 93] 
	# BATAS UPPER RANGE ATAU TERANG
	u = [173, 194, 128]

	mask = masking(img, l, u)

	imgrect, c, a, r = rectangle(mask, img)

	# read('', rect)

	return imgrect, c, a, r

if __name__ == '__main__':

	dirpath = '../Data/Stadium/test/*'
	path = glob.glob(dirpath)

	i = 0
	for item in path:
		i += 1
		# if i > 3:
		# 	break

		img  = cv2.imread(item, cv2.IMREAD_COLOR)
		# read('',img, i)

		r, c, _, _ = segmentation(img)
		cv2.imwrite(f"../Data/coba/imgcrop_{i}.jpg", r)
		
		im_rgb = cv2.cvtColor(r, cv2.COLOR_RGB2LAB)
		read('', im_rgb)
		cv2.imwrite(f"../Data/coba/imglab_{i}.jpg", im_rgb)
		# print(c)
		

		# filename = Path(item).stem

		# imgres, c, a = segmentation(img)

		# if c == 0:
		# 	print('\t',filename, 'not detected')
		# 	# cv2.imwrite(f"../Data/Dataset/crop/Uninfected/img_{i}.jpg", imgres)
		# else:
		# 	print('copying', filename)
		# 	# cv2.imwrite(f"../Data/Dataset/crop/Parasitized/img_{i}.jpg", imgres)

'''

'''