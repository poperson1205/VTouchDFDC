import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
import json
from PIL import Image

DIR = '../deepfake-detection-challenge/train_sample_videos'
OUTDIR = './train_sample_videos_diff'

meta_path = os.path.join(DIR, 'metadata.json')
with open(meta_path) as f:
    meta_json = json.load(f)
    plt.ion()
    frame_counts = []
    for video_name, att in meta_json.items():
        if not ('original' in att):
            continue
        if att['original'] is None:
            continue

        video_path = os.path.join(DIR, video_name)
        video_path_original = os.path.join(DIR, att['original'])
        capture_image = cv2.VideoCapture(video_path)
        if not capture_image.isOpened():
            continue

        frame_count = capture_image.get(cv2.CAP_PROP_FRAME_COUNT)
        print('%s: %d' % (video_name, frame_count))

        frame_counts.append(frame_count)

    df = pd.DataFrame({'name': video_name, 'frame_count': frame_counts})
    df.to_csv('frame_count.csv')
