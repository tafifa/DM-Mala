import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show(img):
	cv2.imshow("test img", img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

def read(dir):
  img = cv2.imread(dir)
  im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  imgplot = plt.imshow(im_rgb)
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
  imgplot = plt.imshow(segmented_image)
  plt.show()
  cv2.imwrite('segm.png', segmented_image)

def rectangle(dir):
  img1 = cv2.imread('segm.png')
  img2 = cv2.imread(dir)
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
      if M['m00'] != 0.0 and area > 20:
          x1 = int(M['m10'] / M['m00'])
          y1 = int(M['m01'] / M['m00'])
          print(f'Area of contour {i + 1}:', area)
          # print(x1, ' ', y1)

          # print(f'Perimeter of contour {i + 1}:', perimeter)
          img1 = cv2.drawContours(img2, [cnt], -1, (0, 255, 255), 1)
          x, y, w, h = cv2.boundingRect(cnt)

          # Print the coordinates
          print(f"Object at ({x}, {y}) ({x+w}, {y+h})")

          xmin.append(x)
          xmax.append(x+w)
          ymin.append(y)
          ymax.append(y+h)

          # Draw the bounding rectangle on the segmentation image
          # cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 1)

          # cv2.putText(img2, f'{i + 1}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
          # cv2.putText(img1, f'Area: {area}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
          # cv2.putText(img1, f'Perimeter: {perimeter}', (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
  cv2.rectangle(img2, (min(xmin), min(ymin)), (max(xmax), max(ymax)), (0, 255, 0), 1)
  read(img2)


if __name__ == '__main__':
  dir = 'image2.png'

  # l = [125, 6, 88]
  # u = [170, 101, 156]
  l = [133, 19, 104]
  u = [211, 114, 165]

  # read(dir)
  segmentation(dir, l, u)
  # rectangle(dir)

  # dirpath = 'Data\Dataset\Sizon\*'
  # path = glob.glob(dirpath)
  # for item in path:
  #   segmentation(item, l, u)
  #   rectangle3(item)
    