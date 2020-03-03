import os
import json
import sys
import cv2

from tqdm import tqdm

DATA_ROOT = '/media/vtouchinc02/database/RawData/deepfake'
OUTPUT_ROOT = '/media/vtouchinc02/database/RawData/deepfake-faces-retinaface'

folder_name = 'dfdc_train_part_46'
video_name = 'wddsnkeghb.mp4'
video_path = os.path.join(DATA_ROOT, folder_name + '/' + video_name)
capture = cv2.VideoCapture(video_path)
while (True):
    ret, frame = capture.read()
    if ret == False:
        break

    cv2.imshow(video_name, frame)
    cv2.waitKey(10)
