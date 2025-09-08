# imports
import numpy as np
import argparse
import cv2
import imutils
from skimage.filters import threshold_local
from transform import *

NEW_IMAGE_SIZE = 500
EPS_FACTORS = [0.02, 0.03, 0.04, 0.05]
DOCUMENT_NBR_POINTS = 4

# enabling CLI argument, no hardcoded paths needed
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# load image, clone, compute ratio and resize
image = cv2.imread(args["image"])
if image is None:
    raise FileNotFoundError(f"Could not load image: {args['image']}")
original = image.copy()
ratio = image.shape[0] / float(NEW_IMAGE_SIZE)
image = imutils.resize(image, height=NEW_IMAGE_SIZE)

# convert to grayscale, Gaussian Blur, Canny-Edge Detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.GaussianBlur(gray_image, (5,5), 0)
edge_image = cv2.Canny(gray_image, 75, 200)

# find all contours based on copy of edge_image, take the 5 largest based on area
contours = cv2.findContours(edge_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

document_contour = None

for c in contours:
    for eps in EPS_FACTORS:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps*perimeter, True)
        
        if len(approx) == DOCUMENT_NBR_POINTS:
            document_contour = approx
            break
    
    if document_contour is not None:
        break

cv2.drawContours(image, [document_contour], -1, (0, 0, 255), 2)

warped_image = four_point_transform(original, document_contour.reshape(4, 2)*ratio)
warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
threshold = threshold_local(warped_image, 11, offset=10, method="gaussian")
warped_image = (warped_image > threshold).astype("uint8")*255

cv2.imshow("Original", imutils.resize(original, height=650))
cv2.imshow("Scanned", imutils.resize(warped_image, height=650))

cv2.waitKey(0)
cv2.destroyAllWindows()

