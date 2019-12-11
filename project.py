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

# apply opening
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# detect contours in the edge map
cnts = cv2.findContours(opened.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1]

# initialize a list for objects areas
roi = np.zeros(opened.shape, dtype="uint8")
shapeArea = []
# loop over the contours
if len(cnts) > 0:
    for c in cnts:
        shapeArea.append(cv2.contourArea(c))

    # grab the index of biggest object
    idx = np.argmax(np.asarray(shapeArea))
    # extract the bounding box ROI from the mask
    (x, y, w, h) = cv2.boundingRect(cnts[idx])
    roi[y:y + h, x:x + w] = opened[y:y + h, x:x + w]
    # roiImage = image[y:y + h, x:x + w]
segmentedImage = cv2.bitwise_and(image, image, mask=roi)

cv2.imshow("Final Image", segmentedImage)
cv2.waitKey(0)
cv2.imwrite("out/{}/segmentation.jpg".format(group), segmentedImage)

#####################################################################################