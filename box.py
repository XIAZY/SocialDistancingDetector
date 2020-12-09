import numpy as np
import cv2 as cv

# adapted from https://docs.opencv.org/master/db/d5b/tutorial_py_mouse_handling.html

class Box:
  def draw_circle(self, event, x, y, flags, param):
      if event == cv.EVENT_LBUTTONDBLCLK:
        self.points.append((x, y))
        cv.circle(img, (x, y), 10, (255, 0, 0), -1)

  def box(self, img):
    # return four points drawn by the user, along with the modified image
    self.points = []

    cv.namedWindow('image')
    cv.setMouseCallback('image', self.draw_circle)

    while len(self.points) < 4:
      cv.imshow('image', img)
      k = cv.waitKey(20) & 0xFF
    
    return self.points, img

def boxsort(points):
  if len(points) != 4:
    return points

  top2bottom = sorted(points, key=lambda x: x[1])
  tltr = sorted(top2bottom[:2], key=lambda x: x[0])
  blbr = sorted(top2bottom[2:], key=lambda x: x[0])

  return blbr + tltr

img = cv.imread('images/crosswalk.jpg', cv.IMREAD_GRAYSCALE)

box = Box()
points, mod = box.box(img)

# birdseye(np.float32(boxsort(points)), img)
