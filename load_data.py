import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
import json
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

mtcnn = MTCNN()

DIR = '/workspace/code/dfdc_train_part_0'

meta_path = os.path.join(DIR, 'metadata.json')
with open(meta_path) as f:
    meta_json = json.load(f)
    plt.ion()
    for video_name, att in meta_json.items():
        print(video_name)
        video_path = os.path.join(DIR, video_name)
        capture_image = cv2.VideoCapture(video_path)
        while True:
            ret, frame = capture_image.read()
            if ret is False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            face = mtcnn(frame)
            if face is None:
                continue

            face = np.transpose(face, (1, 2, 0))
            face = np.array(face)
            face = (face + 1.0)/2.0

            plt.clf()
            plt.imshow(face)
            plt.pause(.1)
