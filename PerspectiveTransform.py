import numpy as np
import cv2 as cv

from detection import default_dedector, draw_boxes
from box import Box

def birdseye_matrix(pts, img):
    def boxsort(points):
        top2bottom = sorted(points, key=lambda x: x[1])
        tltr = sorted(top2bottom[:2], key=lambda x: x[0])
        blbr = sorted(top2bottom[2:], key=lambda x: x[0])

        return blbr + tltr
    
    pts = np.float32(boxsort(pts))

    length = img.shape[0]
    width = img.shape[1]
    
    new = np.float32([[0, length], [width, length], [0, 0], [width, 0]])
    
    return cv.getPerspectiveTransform(pts, new)

def transform_img(M, img):
    return cv.warpPerspective(img, M, (img.shape[1], img.shape[0]))

def transform_persons(M, persons):
    centers = []

    for person in persons:
        left, top, w, h = person[2]
        centers.append([left + w/2, top + h/2])
    
    centers = np.array(centers)
    centers = np.float32(centers).reshape(-1, 1, 2)

    return cv.perspectiveTransform(centers, M).reshape(-1, 2)

def draw_circles(centers, img):
    for center in centers:
        x, y = center
        cv.circle(img, (x, y), 10, (0, 255, 0), -1)

if __name__ == '__main__':
    img = cv.imread('images/street.png')

    detector = default_dedector()
    persons = detector.get_persons(img)

    box = Box()
    points, mod = box.box(img)

    M = birdseye_matrix(points, img)

    transformed_img = transform_img(M, img)
    transformed_persons = transform_persons(M, persons)
    
    draw_circles(transformed_persons, transformed_img)
    cv.imshow("transformed", transformed_img)

    cv.waitKey()
