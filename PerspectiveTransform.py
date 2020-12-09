import numpy as np
import cv2 as cv

from detection import default_dedector, draw_boxes
from box import Box

def boxsort(points):
    top2bottom = sorted(points, key=lambda x: x[1])
    tltr = sorted(top2bottom[:2], key=lambda x: x[0])
    blbr = sorted(top2bottom[2:], key=lambda x: x[0])

    return blbr + tltr

def birdseye_matrix(pts, img):
    pts = np.float32(boxsort(pts))

    # preserve a "horizontal" side lengths
    length = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
    new = np.float32([[0, length], [length, length], [0, 0], [length, 0]])

    # transformation to birdseye view
    M = cv.getPerspectiveTransform(pts, new)

    # - - -

    h, w = img.shape[:2]
    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

    transformed_corners = cv.perspectiveTransform(corners, M).reshape(-1, 2)

    x_max, y_max = np.max(transformed_corners, axis=0)
    x_min, y_min = np.min(transformed_corners, axis=0)

    # translation so all pixels are positive
    t = [-x_min, -y_min]
    t_M = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    # size so all pixels fit in frame
    s = (x_max - x_min, y_max - y_min)
    
    return t_M @ M, s

def transform_persons(M, persons):
    shoes = []

    for person in persons:
        left, top, w, h = person[2]
        shoes.append([left + w/2, top + h])
    
    shoes = np.array(shoes)
    shoes = np.float32(shoes).reshape(-1, 1, 2)

    return cv.perspectiveTransform(shoes, M).reshape(-1, 2)

def draw_circles(shoes, img):
    for x, y in shoes:
        cv.circle(img, (x, y), 10, (0, 255, 0), -1)

if __name__ == '__main__':
    img = cv.imread('images/street.png')

    detector = default_dedector()
    persons = detector.get_persons(img)

    box = Box()
    points, mod = box.box(img)

    M, s = birdseye_matrix(points, img)

    transformed_img = cv.warpPerspective(img, M, s)

    transformed_persons = transform_persons(M, persons)
    draw_circles(transformed_persons, transformed_img)

    transformed_img = cv.resize(transformed_img, (1000, int(1000 / s[0] * s[1])))

    cv.imshow("transformed", transformed_img)
    cv.waitKey()
