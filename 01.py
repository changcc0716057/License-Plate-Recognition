import numpy as np
import cv2

img_path = 'C:/Users/JulieTai/Desktop/Julie/visualstudiocode/.vscode/2021_spring/License_Plate_Recognition/'
img = cv2.imread(img_path + 'car2.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_img', gray_img)
cv2.waitKey(0)
#cv2.imwrite('C:/Users/JulieTai/Desktop/Julie/visualstudiocode/.vscode/2021_spring/License_Plate_Recognition/car1_gray.jpg', gray_img)

# Gaussian smoothing filter
Gaussian_img = cv2.GaussianBlur(gray_img, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
# median filter
median_img = cv2.medianBlur(Gaussian_img, 5)
cv2.imshow('blur_img', median_img)
cv2.waitKey(0)
#cv2.imwrite('C:/Users/JulieTai/Desktop/Julie/visualstudiocode/.vscode/2021_spring/License_Plate_Recognition/car1_blur_median.jpg', gray_img)

# Sobel: gradient in the x and y direction
sobel_x = cv2.Sobel(median_img, cv2.CV_16S, 1, 0, ksize = 3)
sobel_y = cv2.Sobel(median_img, cv2.CV_16S, 0, 1, ksize = 3)
# change the type into uint8
abs_x = cv2.convertScaleAbs(sobel_x)
abs_y = cv2.convertScaleAbs(sobel_y)
Sobel_img = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
#cv2.imshow('x', abs_x)
#cv2.imwrite('C:/Users/JulieTai/Desktop/Julie/visualstudiocode/.vscode/2021_spring/License_Plate_Recognition/car1_sobel_x.jpg', abs_x)
#cv2.imshow('y', abs_y)
#cv2.imwrite('C:/Users/JulieTai/Desktop/Julie/visualstudiocode/.vscode/2021_spring/License_Plate_Recognition/car1_sobel_y.jpg', abs_y)
cv2.imshow('sb', Sobel_img)
cv2.waitKey(0)
#cv2.imwrite('C:/Users/JulieTai/Desktop/Julie/visualstudiocode/.vscode/2021_spring/License_Plate_Recognition/car1_sobel.jpg', Sobel_img)

#二值化處理 周圍畫素影響
#再進行一次高斯去噪
blurred_img = cv2.GaussianBlur(Sobel_img, (9, 9), 0)
ret, binary_img = cv2.threshold(blurred_img , 70, 255, cv2.THRESH_BINARY)
cv2.imshow('binary', binary_img)
cv2.waitKey(0)
#cv2.imwrite('C:/Users/JulieTai/Desktop/Julie/visualstudiocode/.vscode/2021_spring/License_Plate_Recognition/car1_binary.jpg', binary_img)

"""
# Dilation and erosion operations
# construct a 9x1 and a 9x7 structuring elements
element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
# 膨脹一次，讓輪廓突出
dilation_img = cv2.dilate(binary_img, element2, iterations = 1)
#cv2.imshow('Dilation1', dilation_img)
#cv2.waitKey(0)
# 腐蝕一次，去掉細節
erosion_img = cv2.erode(dilation_img, element1, iterations = 1)
#cv2.imshow('erosion', erosion_img)
#cv2.waitKey(0)
# 再次膨脹，讓輪廓明顯一些
dilation_img_2 = cv2.dilate(erosion_img, element2, iterations = 1)
#cv2.imshow('Dilation2', dilation_img_2)
#cv2.waitKey(0)
"""

#建立一個橢圓核函式
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
#執行影象形態學
closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
closed_img = cv2.erode(closed_img, None, iterations=3)
closed_img = cv2.dilate(closed_img, None, iterations=3)
cv2.imshow('erode dilate', closed_img)
cv2.waitKey(0)

cnts, _ = cv2.findContours(closed_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#畫出輪廓
# Sort by area, descending
sort_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
# Largest area
largest_cnt = sort_cnts[0]

#compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(largest_cnt)
#print('rectt', rect)
box_points = np.int0(cv2.boxPoints(rect))
#print('Box', box_points)

#draw a bounding box arounded the detected barcode and display the image
final_img = cv2.drawContours(img.copy(), [box_points], -1, (0, 0, 255), 3)

cv2.imshow('final_img', final_img)
cv2.waitKey(0)

cv2.destroyAllWindows()