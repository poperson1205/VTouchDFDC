import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
import json

DIR = '/workspace/code/dfdc_train_part_0'

meta_path = os.path.join(DIR, 'metadata.json')
with open(meta_path) as f:
    meta_json = json.load(f)
    plt.ion()
    for video_name, att in meta_json.items():
        print(video_name)
        video_path = os.path.join(DIR, video_name)
        capture_image = cv2.VideoCapture(video_path)
        count = 0
        while True:
            print(count)
            count += 1
            ret, frame = capture_image.read()
            if ret is False:
                break

            # Test
            plt.clf()
            plt.axis('off')
            plt.imshow(frame)
            plt.pause(.1)
