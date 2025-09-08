import numpy as np
import cv2

def order_points(pts):
    rectangle = np.zeros((4, 2), dtype="float32")
    
    sum_points = pts.sum(axis = 1)
    diff_points = np.diff(pts, axis=1)
    
    rectangle[0] = pts[np.argmin(sum_points)]
    rectangle[1] = pts[np.argmin(diff_points)]
    rectangle[2] = pts[np.argmax(sum_points)]
    rectangle[3] = pts[np.argmax(diff_points)]
    
    return rectangle
    
def four_point_transform(image, pts):
    rectangle = order_points(pts)
    (tl, tr, br ,bl) = rectangle

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

    transform_matrix = cv2.getPerspectiveTransform(rectangle, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (maxWidth, maxHeight))

    return warped

