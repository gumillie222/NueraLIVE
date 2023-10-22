import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('SHSY5Y_Phase_B10_1_00d04h00m_2.tif')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv.threshold(gray, 130, 255, cv.THRESH_BINARY)
cv.imshow('Grayscaled image', thresh)

# Create a copy of the thresholded image to draw contours on
contour_img = thresh.copy()

contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

circle = 0
ellipse = 0

for contour in contours:
    # Approximate the contour
    epsilon = 0.04 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)

    # Calculate the aspect ratio
    x, y, _, _ = cv.boundingRect(contour)
    
    if (len(approx) < 6 and len(approx) > 3):
        shape = 'Ellipse'
        ellipse += 1
        cv.drawContours(img, [contour], 0, (0, 0, 255), 2)
        cv.putText(img, shape, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
    elif len(approx) > 5:
        shape = 'Circle'
        circle += 1
        cv.drawContours(img, [contour], 0, (0, 255, 0), 2)
        cv.putText(img, shape, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

print(int(circle/3), "circles")
print(int(ellipse/3), "ellipses")

# Display the result
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()