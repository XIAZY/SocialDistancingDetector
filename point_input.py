import numpy as np
import cv2 as cv

# adapted from https://docs.opencv.org/master/db/d5b/tutorial_py_mouse_handling.html

class PointInput:
  def draw_circle(self, event, x, y, flags, param):
      if event == cv.EVENT_LBUTTONDBLCLK:
        self.points.append((x, y))
        cv.circle(self.img, (x, y), 10, (255, 0, 0), -1)

  def n_points(self, n, img):
    # return four points drawn by the user, along with the modified image
    self.points = []
    self.img = img.copy()
    
    cv.namedWindow('image')
    cv.setMouseCallback('image', self.draw_circle)

    while len(self.points) < n:
      cv.imshow('image', self.img)
      k = cv.waitKey(20) & 0xFF
    
    return self.points, self.img
