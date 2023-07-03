import numpy as np
import cv2

#UNGU
lower = np.array([165, 75, 250])
upper = np.array([175, 85, 255])

mask = cv2.inRange(hsv, lower, upper)
#FIX
num = 1
array_class = []
rect_class = []
areas_mask=[]
for (i,contour) in enumerate(contours):	
    if 10.0 < cv2.contourArea(contour) < 500:
        rect = cv2.boundingRect(contour)#area persegi object
        areas = cv2.contourArea(contour)#area  object (lebih polygon)
        areas_mask.append(areas)
        #print(areas)
        #print(contour)
        #print(rect)
        #cv2.drawContours(im, contour, -1, (0, 255, 255), 3)
        area = rect[2] * rect[3]
        #print(str([rect[2],rect[3]])+"="+str(area))
        if area>150:
            cv2.drawContours(original, contour, -1, (0, 255, 255), 3)
            #print("Lebar x Panjang"+str([rect[2],rect[3]])+"="+str(area))]
            #print(areas)
            #print(contour)#nilai array bisa digunakan untuk cropping dg polygon
            #print(rect)#nilai array bisa digunakan untuk cropping dg rectangle''
            rect_class.append(contour)
            #CROP PETAK
            x,y,w,h=rect #ganti areas
            #crop_img = original[y:y+h, x:x+w]
            crop_img = croped[y:y+h, x:x+w]
            cv2.imwrite('./output_cnn/cnn-'+str(num)+'.jpg', crop_img)
            #cv2.rectangle(original, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.rectangle(croped, (x, y), (x+w, y+h), (255,0,0), 2)
        num += 1