from detection import get_persons, draw_boxes
import cv2
import numpy as np

def get_violators(persons):
    violators = set()
    n = len(persons)
    avg_h = np.mean([p[2][3] for p in persons])
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
            p = (min(a)/max(a))

            center = [(top[0]+h[0]//2, left[0]+w[0]//2),
                      (top[1]+h[1]//2, left[1]+w[1]//2)]
            euc_dist = np.sqrt(
                (center[1][0]-center[0][0])**2 + (center[1][1]-center[0][1])**2)
            # print(euc_dist)
            inv_dist = euc_dist / (sum(h)/2)
            # inv_dist = euc_dist / avg_h
            prod = inv_dist / p
            print(prod)
            if prod < 1.11:
                # print(euc_dist, inv_dist, prod)
                violators.add(i)
                violators.add(j)
    
    return [persons[i] for i in violators]


if __name__ == '__main__':
    img = cv2.imread('images/street.png')

    persons = get_persons(img)
    violators = get_violators(persons)
    draw_boxes(img, violators)
