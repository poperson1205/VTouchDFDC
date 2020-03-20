import os
from threading import Thread
import copy
import json
import sys
import cv2
import numpy as np

from tqdm import tqdm

INPUT_ROOT = '/media/vtouchinc02/database/RawData/deepfake-32frame'
OUTPUT_ROOT = '/media/vtouchinc02/database/RawData/deepfake-32frame-clean-non-fake-detections'

def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size
    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized

def tag_manipulated_ratio(start_index, end_index):
    for i in range(start_index, end_index):
        folder_name = 'dfdc_train_part_%d' % i
        print(folder_name)

        folder_path = os.path.join(INPUT_ROOT, folder_name)
        metadata_path = os.path.join(INPUT_ROOT, folder_name + '/metadata.json')
        with open(metadata_path) as metadata_fp:
            metadata = json.load(metadata_fp)

        # Clean 'face' in FAKE videos
        for video_name, att in tqdm(metadata.items()):
            if not ('original' in att):
                continue
            if att['original'] is None:
                continue

            video_name_original = att['original']
            att_original = metadata[video_name_original]

            if not ('face' in att_original):
                print('No face detected: %s' % video_name_original)
            
            att['face'] = copy.deepcopy(att_original['face'])
            face_results = att['face']
            for frame_index, frame_result in face_results.items():
                # print(frame_index)
                for box_index, box_result in enumerate(frame_result):
                    # print('Score: %f' % box_result['score'])

                    image_name = '%s-%s-%d.png' % (video_name[:-4], frame_index, box_index)
                    image_name_original = '%s-%s-%d.png' % (video_name_original[:-4], frame_index, box_index)

                    image_path = os.path.join(folder_path, image_name)
                    image_path_original = os.path.join(folder_path, image_name_original)

                    if not os.path.isfile(image_path):
                        continue

                    if not os.path.isfile(image_path_original):
                        continue

                    img = cv2.imread(image_path)
                    img = isotropically_resize_image(img, size=224)
                    img_original = cv2.imread(image_path_original)
                    img_original = isotropically_resize_image(img_original, size=224)

                    img_min, img_max, _, _ = cv2.minMaxLoc(cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY))
                    # print('[Image] min: %d, max: %d' % (img_min, img_max))

                    img_diff = cv2.absdiff(img, img_original)
                    img_diff = cv2.cvtColor(img_diff, cv2.COLOR_RGB2GRAY)
                    diff_min, diff_max, _, __ = cv2.minMaxLoc(img_diff)
                    # print('[Diff] min: %d, max: %d' % (diff_min, diff_max))

                    _, img_diff_thresholded = cv2.threshold(img_diff, 16, maxval=255, type=cv2.THRESH_BINARY)
                    # img_diff_thresholded = cv2.adaptiveThreshold(img_diff,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,5,1)
                    # img_diff_thresholded_opened = cv2.morphologyEx(img_diff_thresholded, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
                    
                    box_result['manipulated_ratio'] = cv2.countNonZero(img_diff_thresholded) / (224*224)
                    box_result['img_min'] = img_min
                    box_result['img_max'] = img_max
                    box_result['diff_min'] = diff_min
                    box_result['diff_max'] = diff_max

                    # if cv2.countNonZero(img_diff_thresholded) < (224*224/64):
                    #     cv2.putText(img, 'NON-FAKE', (14, 56), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=4)

                    # cv2.imshow('original', cv2.convertScaleAbs(img_original, None, alpha=255.0/img_max, beta=img_min))
                    # cv2.imshow('fake', cv2.convertScaleAbs(img, None, alpha=255.0/img_max, beta=img_min))
                    # cv2.imshow('diff', cv2.normalize(img_diff, None, alpha=0, beta=img_max, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
                    # cv2.imshow('diff_thresholded', img_diff_thresholded)
                    # cv2.imshow('diff_thresholded_opened', img_diff_thresholded_opened)
                    
                    # cv2.waitKey(10)

        output_folder_path = os.path.join(OUTPUT_ROOT, folder_name)
        if not os.path.isdir(output_folder_path):
            os.mkdir(output_folder_path)
        with open(os.path.join(output_folder_path, 'metadata.json'), "w") as fp:
            json.dump(metadata, fp)

if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_ROOT):
        os.mkdir(OUTPUT_ROOT)

    tag_manipulated_ratio(0, 10)

    # threads = []
    # threads.append(Thread(target=tag_manipulated_ratio, args=(0,5)))
    # threads.append(Thread(target=tag_manipulated_ratio, args=(5,10)))
    # # threads.append(Thread(target=tag_manipulated_ratio, args=(10,15)))
    # # threads.append(Thread(target=tag_manipulated_ratio, args=(15,20)))

    # for thread in threads:
    #     thread.start()

    # for thread in threads:
    #     thread.join()