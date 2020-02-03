import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
import json
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

mtcnn = MTCNN(keep_all=True, device='cuda')

DIR = '/workspace/code/dfdc_train_part_0'

meta_path = os.path.join(DIR, 'metadata.json')
with open(meta_path) as f:
    meta_json = json.load(f)
    plt.ion()
    for video_name, att in meta_json.items():
        print(video_name)
        video_path = os.path.join(DIR, video_name)
        capture_image = cv2.VideoCapture(video_path)
        frame_count = 0
        while True:
            ret, cv_img = capture_image.read()
            if ret is False:
                break
            pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            boxes, _ = mtcnn.detect(pil_img)
            if boxes is None:
                continue

            box_count = 0
            for box in boxes:
                box = box.astype(int)
                face_img = cv_img[box[1]:box[3], box[0]:box[2]]
                cv2.imwrite(video_path.rsplit('.')[0] + '-{0}-{1}.jpg'.format(frame_count, box_count), face_img)
                box_count += 1

            frame_count += 1
