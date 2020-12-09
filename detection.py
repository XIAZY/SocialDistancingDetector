import cv2 as cv
import numpy as np

import configparser

# adapted from https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.py


class Detector:
    def init_model(self, config_file, weights_file):
        self.net = cv.dnn.readNetFromDarknet(config_file, weights_file)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

        config = configparser.ConfigParser(strict=False)
        config.read(config_file)

        self.inputShape = (
            int(config['net']['width']), int(config['net']['height']))

    def load_classes(self, classes_file):
        self.classes = open(classes_file).read().strip().split('\n')

    def detect(self, img, conf_thresh, NMS_thresh):
        blob = cv.dnn.blobFromImage(
            img, 1/255.0, self.inputShape, swapRB=True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        h, w = img.shape[:2]

        classes = []
        confs = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                conf = scores[classId]

                if conf > conf_thresh:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    width = int(detection[2] * w)
                    height = int(detection[3] * h)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)

                    classes.append(self.classes[classId])
                    confs.append(float(conf))
                    boxes.append([left, top, width, height])

        nms_indices = cv.dnn.NMSBoxes(
            boxes, confs, conf_thresh, NMS_thresh).flatten()

        classes = [classes[i] for i in nms_indices]
        confs = [confs[i] for i in nms_indices]
        boxes = [boxes[i] for i in nms_indices]

        return zip(classes, confs, boxes)
    
    def get_persons(self, img):
        detected = self.detect(img, 0.5, 0.5)

        return list(filter(lambda obj: obj[0] == 'person', detected))

def draw_boxes(img, persons):
    for person in persons:
        left, top, w, h = person[2]
        cv.rectangle(img, (left, top), (left + w, top + h), (255, 0, 0), 1)

    cv.imshow('boxes', img)
    cv.waitKey()


def default_dedector():
    detector = Detector()
    detector.init_model('yolo/yolov3.cfg', 'yolo/yolov3.weights')
    detector.load_classes('yolo/coco.names')

    return detector


def main():
    img = cv.imread('images/younge.jpg')

    persons = get_persons(img)
    draw_boxes(img, persons)


if __name__ == '__main__':
    main()
