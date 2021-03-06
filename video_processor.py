import cv2
import skvideo.io
from crowd_detector import get_processed_img
from detection import default_dedector
import numpy as np

from PerspectiveTransform import Transformer

def get_process_video(input_file_path, processor):
    detector = default_dedector()

    videogen = skvideo.io.vreader(input_file_path)

    data = []
    for frame in videogen:
        print('.',end='')
        processed_frame = processor(detector, frame)
        data.append(processed_frame)
    
    return data

def write_processed_video(input_file_path, output_file_path, processor):
    data = get_process_video(input_file_path, processor)
    skvideo.io.vwrite(output_file_path, data)

if __name__ == '__main__':
    # write_processed_video('videos/cctv.mp4', 'output.mp4', get_processed_img)

    videogen = skvideo.io.vreader('videos/cctv.mp4')
    frame = next(videogen)

    transformer = Transformer()
    transformer.calibrate(frame)

    transformed_img = transformer.get_processed_image(default_dedector(), frame)

    cv2.imshow("transformed", transformed_img)
    cv2.waitKey()

    write_processed_video('videos/cctv.mp4', 'output.mp4', transformer.get_processed_image)
