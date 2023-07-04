import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def show(img):
	cv2.imshow("test img", img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

def read(dir):
  # img = cv2.imread(dir)
  # im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  imgplot = plt.imshow(dir)
  plt.show()

def segmentation(dir, l, u):
  # Read the image
  image = cv2.imread(dir)

  # Convert the image to the HSV color space
  hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Define the lower and upper threshold values for the color range
  lower_threshold = np.array(l)  # Lower threshold for the color range (in HSV)
  upper_threshold = np.array(u)  # Upper threshold for the color range (in HSV)

  # Create a binary mask by applying the color threshold
  mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)

  # Apply the mask to the original image
  segmented_image = cv2.bitwise_and(image, image, mask=mask)

  # Display the original image and the segmented image
  # imgplot = plt.imshow(segmented_image)
  # plt.show()
  cv2.imwrite('segm.png', segmented_image)

def rectangle(dir):
  import cv2

  img = cv2.imread(dir)
  img_seg = cv2.imread('segm.png')
  gray = cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY)
  # show(gray)
  _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  show(thresh)

  # Find contours in the binary image
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  # show(contours)

  # Get the bounding rectangle coordinates of the largest contour
  x1, y1, w, h = cv2.boundingRect(contours[0])
  x2, y2 = x1 + w, y1 + h

  # Print the top-left and bottom-right coordinates of the bounding rectangle
  print((x1, y1), (x2, y2))

  # Draw the bounding rectangle on the image
  cv2.rectangle(img_seg, (x1, y1), (x2, y2), (0, 255, 0), 2)

  # Save the result image
  cv2.imwrite('res.jpg', img_seg)

def rectangle2(dir):
   # Load image, grayscale, Otsu's threshold
  image = cv2.imread(dir)
  original = image.copy()
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

  # Morph open to remove noise
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
  opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

  # Find contours, obtain bounding box, extract and save ROI
  ROI_number = 0
  cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
      ROI = original[y:y+h, x:x+w]
      cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
      ROI_number += 1

  cv2.imshow('image', image)
  cv2.imshow('thresh', thresh)
  cv2.imshow('opening', opening)
  cv2.waitKey()

def rectangle3():
  img1 = cv2.imread('segm.png')
  img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(img, 10, 255, 0)
  contours, hierarchy = cv2.findContours(thresh, 1, 2)
  print("Number of contours in image:", len(contours))
  xmin, xmax, ymin, ymax = [], [], [], []

  for i, cnt in enumerate(contours):
      M = cv2.moments(cnt)
      area = cv2.contourArea(cnt)
      perimeter = cv2.arcLength(cnt, True)
      perimeter = round(perimeter, 4)
      if M['m00'] != 0.0 and area > 60:
          x1 = int(M['m10'] / M['m00'])
          y1 = int(M['m01'] / M['m00'])
          print(f'Area of contour {i + 1}:', area)
          # print(x1, ' ', y1)

          # print(f'Perimeter of contour {i + 1}:', perimeter)
          img1 = cv2.drawContours(img1, [cnt], -1, (0, 255, 255), 1)
          x, y, w, h = cv2.boundingRect(cnt)

          # Print the coordinates
          print(f"Object at ({x}, {y}) ({x+w}, {y+h})")

          xmin.append(x)
          xmax.append(x+w)
          ymin.append(y)
          ymax.append(y+h)

          # Draw the bounding rectangle on the segmentation image
          # cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 1)

          cv2.putText(img1, f'{i + 1}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
          # cv2.putText(img1, f'Area: {area}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
          # cv2.putText(img1, f'Perimeter: {perimeter}', (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
  cv2.rectangle(img1, (min(xmin), min(ymin)), (max(xmax), max(ymax)), (0, 255, 0), 1)
  read(img1)
  # cv2.imshow("Image", img1)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

if __name__ == '__main__':
  dir = 'Gametosit/C140P101ThinF_IMG_20151005_211530_cell_147.png'

  path = glob.glob(dir)
  l = [125, 6, 88]
  u = [170, 101, 156]
  # read(dir)
  segmentation(dir, l, u)
  rectangle3()
  # rectangle(dir)
  # rectangle2(dir)
  

  # plt.show()