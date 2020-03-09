import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
import json
from PIL import Image
from tqdm import tqdm

DIR = '/media/vtouchinc02/database/RawData/deepfake/dfdc_train_part_30'

def get_num_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    count = 0
    while True:
        if cap.grab():
            count += 1
        else:
            break

    return count

meta_path = os.path.join(DIR, 'metadata.json')
with open(meta_path) as f:
    meta_json = json.load(f)
    plt.ion()
    for video_name, att in tqdm(meta_json.items()):
        if not ('original' in att):
            continue
        if att['original'] is None:
            continue

        video_path = os.path.join(DIR, video_name)
        video_name_original = att['original']
        video_path_original = os.path.join(DIR, video_name_original)
        capture_image = cv2.VideoCapture(video_path)
        if not capture_image.isOpened():
            continue
        capture_image_original = cv2.VideoCapture(video_path_original)
        if not capture_image_original.isOpened():
            continue
        
        # num_frames = get_num_frames(video_path)
        # num_frames_original = get_num_frames(video_path_original)

        # print('frames: %d' % num_frames)
        # print('original frames: %d' % num_frames_original)

        frame_count = 0
        while True:
            print (frame_count)
            ret, cv_img = capture_image.read()
            if ret is False:
                break

            ret_original, cv_img_original = capture_image_original.read()
            if ret_original is False:
                break

            cv_diff =  cv2.absdiff(cv_img, cv_img_original) * 4
            cv_diff = cv2.cvtColor(cv_diff, cv2.COLOR_RGB2GRAY)
            # cv_diff = cv2.normalize(cv_diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # _, cv_diff = cv2.threshold(cv_diff, 64, maxval=255, type=cv2.THRESH_BINARY)

            cv2.imshow('original', cv2.resize(cv_img_original, None, None, 0.3, 0.3))
            cv2.imshow('fake', cv2.resize(cv_img, None, None, 0.3, 0.3))
            cv2.imshow('diff', cv2.resize(cv_diff, None, None, 0.3, 0.3))
            cv2.waitKey()
            # cv2.waitKey(delay=1)
            frame_count += 1