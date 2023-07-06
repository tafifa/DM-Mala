import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def read(dir, img):
  if dir != '':
    img = cv2.imread(dir)
    im_rgb = cv2.cvtColor(dir, cv2.COLOR_BGR2RGB)
  else:
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  imgplot = plt.imshow(im_rgb)
  plt.show()

def masking(img, l, u):

  # Convert the image to the HSV color space
  hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # Define the lower and upper threshold values for the color range
  lower_threshold = np.array(l)  # Lower threshold for the color range (in HSV)
  upper_threshold = np.array(u)  # Upper threshold for the color range (in HSV)

  # Create a binary mask by applying the color threshold
  mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)

  # Apply the mask to the original image
  segmented_image = cv2.bitwise_and(img, img, mask=mask)
  
  return segmented_image

def rectangle(mask, imged):

  img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(img, 10, 255, 0)
  contours, hierarchy = cv2.findContours(thresh, 1, 2)

  xmin, xmax, ymin, ymax = [], [], [], []

  if len(contours) < 1:
    return imged
  
  count = 0
  for i, cnt in enumerate(contours):
      area = cv2.contourArea(cnt)

      M = cv2.moments(cnt)
      if M['m00'] != 0.0 and area > 80:

          x, y, w, h = cv2.boundingRect(cnt)

          xmin.append(x)
          xmax.append(x+w)
          ymin.append(y)
          ymax.append(y+h)

          count += 1

  if count > 0:
    crop = imged[min(ymin):max(ymax), min(xmin):max(xmax)]
    # read('', crop)

    return imged
  
  else:
    return imged

def segmentation(img):
  l = [101, 6, 88] # BATAS LOWER RANGE ATAU GELAP
  u = [211, 114, 165] # BATAS UPPER RANGE ATAU TERANG

  mask = masking(img, l, u)

  rect = rectangle(mask, img)

  return rect

if __name__ == '__main__':

  dirpath = '../Data/test/*'
  path = glob.glob(dirpath)
  for item in path:
    img  = cv2.imread(item, cv2.IMREAD_COLOR)
    # read('', img)
    segmentation(img)