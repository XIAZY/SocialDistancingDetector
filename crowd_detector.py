from detection import default_dedector, draw_boxes
import cv2
import numpy as np

def get_violators(persons):
    violators = set()
    n = len(persons)
    for i in range(n):
        for j in range(i+1, n):
            person = [ persons[i], persons[j] ]

            left_i, top_i, w_i, h_i = person[0][2]
            left_j, top_j, w_j, h_j = person[1][2]

            left = [left_i, left_j]
            top = [top_i, top_j]
            w = [w_i, w_j]
            h = [h_i, h_j]
            
            a = [w[0] * h[0], w[1] * h[1]]
            p = (min(h)/max(h))

            center = [(top[0]+h[0]//2, left[0]+w[0]//2),
                      (top[1]+h[1]//2, left[1]+w[1]//2)]
            euc_dist = np.sqrt(
                (center[1][0]-center[0][0])**2 + (center[1][1]-center[0][1])**2)
            inv_dist = euc_dist / (sum(h)/2)
            prod = inv_dist / p

            if prod < 1.11:
                violators.add(i)
                violators.add(j)
    
    return [persons[i] for i in violators]

def get_processed_img(detector, img):
    persons = detector.get_persons(img)
    violators = get_violators(persons)
    img = draw_boxes(img, violators)
    return img

if __name__ == '__main__':
    img = cv2.imread('images/street.png')

    detector = default_dedector()
    persons = detector.get_persons(img)
    violators = get_violators(persons)
    img = draw_boxes(img, violators)
    cv2.imshow('boxes', img)
    cv2.waitKey()
