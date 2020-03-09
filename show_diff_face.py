import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
import json
from PIL import Image

DIR = '/media/vtouchinc02/database/RawData/deepfake-faces-retinaface-json-backup-clean-fake-results-images/dfdc_train_part_0'

meta_path = os.path.join(DIR, 'metadata.json')
with open(meta_path) as f:
    meta_json = json.load(f)
    plt.ion()
    for video_name, att in meta_json.items():
        if not ('original' in att):
            continue
        if att['original'] is None:
            continue

        video_name_original = att['original']
        att_original = meta_json[video_name_original]

        if not ('face' in att_original):
            print('No face detected: %s' % video_name_original)
        
        face_results = att_original['face']
        for frame_index, frame_result in face_results.items():
            print(frame_index)
            for box_index, box_result in enumerate(frame_result):
                print('Score: %f' % box_result['score'])

                image_name = '%s-%s-%d.png' % (video_name[:-4], frame_index, box_index)
                image_name_original = '%s-%s-%d.png' % (video_name_original[:-4], frame_index, box_index)
                img = cv2.imread(os.path.join(DIR, image_name))
                img_original = cv2.imread(os.path.join(DIR, image_name_original))

                img_diff =  cv2.absdiff(img, img_original)
                img_diff = cv2.cvtColor(img_diff, cv2.COLOR_RGB2GRAY)
                # img_diff = cv2.normalize(img_diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                _, img_diff_thresholded = cv2.threshold(img_diff, 8, maxval=255, type=cv2.THRESH_BINARY)
                img_diff_thresholded_opened = cv2.morphologyEx(img_diff_thresholded, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))

                cv2.imshow('original', img_original)
                cv2.imshow('fake', img)
                cv2.imshow('diff', img_diff)
                cv2.imshow('diff_thresholded', img_diff_thresholded)
                cv2.imshow('diff_thresholded_opened', img_diff_thresholded_opened)

                # cv2.imshow('original', cv2.resize(img_original, None, None, 0.3, 0.3))
                # cv2.imshow('fake', cv2.resize(img, None, None, 0.3, 0.3))
                # cv2.imshow('diff', cv2.resize(img_diff, None, None, 0.3, 0.3))
                cv2.waitKey(1)

                if cv2.countNonZero(img_diff_thresholded_opened) == 0:
                    print('Zero!!!')
                    cv2.waitKey()
