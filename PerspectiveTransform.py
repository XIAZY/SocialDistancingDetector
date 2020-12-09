import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import configparser
from detection import get_persons, draw_boxes

def birdseye(pts, img):
    """
    point orientation goes in order of bottom left, bottom right, top left, top right
    """
    length = img.shape[0]
    width = img.shape[1]
    
    new = np.float32([[0, length], [width, length], [0, 0], [width, 0]])
    
    M = cv.getPerspectiveTransform(pts, new)
    
    newImg = cv.warpPerspective(img, M, (width, length))
    
    plt.figure(figsize=(10,10))
    
    plt.imshow(cv.cvtColor(newImg, cv.COLOR_BGR2RGB))
    plt.show()
    
    return M, newImg


def location(pts, img):

    M, transformedImg = birdseye(pts, img)
    persons = get_persons(img)
    
    # dimensions of the trapzoid. Not done yet, just took the box created by
    # the shorter ends.
    maxH = pts[0][1]
    minH = pts[2][1]
    minL = pts[2][0]
    maxL = pts[3][0]
    

    #find the center of person
    center = []
    for person in persons:
        left, top, w, h = person[2]
        pt = [left + w/2, top + h/2]
     
        #test for now
        if pt[0] > 100 and pt[0] < 900 and pt[1] > 400:
            center.append(pt)
    
    center = np.array(center)
    transformedCenter = np.float32(center).reshape(-1, 1, 2)
    transformedCenter = cv.perspectiveTransform(transformedCenter, M)
    
    transformedPoints = []
    for i in range(0, transformedCenter.shape[0]):
        transformedPoints.append([transformedCenter[i][0][0], transformedCenter[i][0][1]])
    transformedPoints = np.array(transformedPoints)
    
    return transformedPoints
