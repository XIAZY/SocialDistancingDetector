import numpy as np
import cv2 as cv

from detection import default_dedector, draw_boxes
from point_input import PointInput

import ctypes

def boxsort(points):
    top2bottom = sorted(points, key=lambda x: x[1])
    tltr = sorted(top2bottom[:2], key=lambda x: x[0])
    blbr = sorted(top2bottom[2:], key=lambda x: x[0])

    return blbr + tltr

def birdseye_matrix(pts, img):
    pts = np.float32(boxsort(pts))

    # preserve a "horizontal" side length
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

def transform_persons(persons, M):
    shoes = []

    for person in persons:
        left, top, w, h = person[2]
        shoes.append([left + w/2, top + h])
    
    shoes = np.array(shoes)
    shoes = np.float32(shoes).reshape(-1, 1, 2)

    return cv.perspectiveTransform(shoes, M).reshape(-1, 2)

def get_violators(transformed_persons, M, sixfeet):
    transformed_sixfeet = cv.perspectiveTransform(np.float32(sixfeet).reshape(-1, 1, 2), M).reshape(-1, 2)
    threshold_dist = np.linalg.norm(transformed_sixfeet[0] - transformed_sixfeet[1])

    violators = set()

    for p1 in transformed_persons:
        for p2 in transformed_persons:
            if tuple(p1) != tuple(p2) and np.linalg.norm(p1 - p2) <= threshold_dist:
                violators.add(tuple(p1))
                violators.add(tuple(p2))
    
    good_citizens = set(tuple(p) for p in transformed_persons)
    good_citizens = good_citizens - violators

    return violators, good_citizens

def draw_circles(coords, img, color):
    for x, y in coords:
        cv.circle(img, (x, y), 10, color, -1)

def reasonable_size(img):
    s = img.shape[:2]
    side_length = 1000

    option1 = (side_length, int(side_length / s[0] * s[1]))
    option2 = (int(side_length / s[1] * s[0]), side_length)

    return cv.resize(img, option1) if option1[1] < option2[0] else cv.resize(img, option2)

if __name__ == '__main__':
    img = cv.imread('images/street.png')

    detector = default_dedector()
    persons = detector.get_persons(img)

    ctypes.windll.user32.MessageBoxW(0, "please draw four points that would look like a square from bird\'s-eye view'", "prompt", 0)
    point_input = PointInput()
    square, _ = point_input.n_points(4, img)

    ctypes.windll.user32.MessageBoxW(0, "please draw two points six feet apart'", "prompt", 0)
    sixfeet, _ = point_input.n_points(2, img)

    M, s = birdseye_matrix(square, img)

    transformed_img = cv.warpPerspective(img, M, s)
    transformed_persons = transform_persons(persons, M)

    violators, good_citizens = get_violators(transformed_persons, M, sixfeet)
    draw_circles(violators, transformed_img, (0, 0, 255))
    draw_circles(good_citizens, transformed_img, (0, 255, 0))

    transformed_img = reasonable_size(transformed_img)

    cv.imshow("transformed", transformed_img)
    cv.waitKey()
