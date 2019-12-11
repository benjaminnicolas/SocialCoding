#####################################################################################
# original template code
import cv2
import numpy as np

group = "group1"


image = cv2.imread("images/{}.jpg".format(group))

cv2.imshow("image", image)
cv2.waitKey(0)


# convert to gray-scale
grayInpaintedBlack = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(_, thresh) = cv2.threshold(grayInpaintedBlack, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#####################################################################################