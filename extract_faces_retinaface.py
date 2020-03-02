import os
import json
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

import sys
import retina_face_detector

from tqdm import tqdm

DATA_ROOT = '/media/vtouchinc02/database/RawData/deepfake'
OUTPUT_ROOT = '/media/vtouchinc02/database/RawData/deepfake-faces-retinaface'

def get_frames(video_path, num_frames=17):
    capture = cv2.VideoCapture(video_path)

    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_idxs = np.linspace(0, frame_count-1, num_frames, endpoint=True, dtype=np.int)

    frames = []
    idxs_read = []
    for frame_idx in range(frame_idxs[0], frame_idxs[-1] + 1):
        ret = capture.grab()
        if not ret:
            print("Error grabbing frame %d from movie %s" % (frame_idx, video_path))
            break
        current = len(idxs_read)
        if frame_idx == frame_idxs[current]:
            ret, frame = capture.retrieve()
            if not ret or frame is None:
                print("Error retrieving frame %d from movie %s" % (frame_idx, video_path))
                break
            frames.append(frame)
            idxs_read.append(frame_idx)

    return idxs_read, frames

if __name__ == '__main__':
    image_size = 224
    gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    detector = retina_face_detector.RetinaFaceDetector(
        network='mobile0.25',
        trained_model='weights/mobilenet0.25_Final.pth',
        vis_thres=0.9,
        show_image=False)

    if not os.path.isdir(OUTPUT_ROOT):
        os.mkdir(OUTPUT_ROOT)

    for folder_name in tqdm(os.listdir(DATA_ROOT)):
        print(folder_name)
        metadata_path = os.path.join(DATA_ROOT, folder_name + '/metadata.json')
        with open(metadata_path) as metadata_fp:
            metadata = json.load(metadata_fp)

        output_folder = os.path.join(OUTPUT_ROOT, folder_name)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        
        for video_name, attributes in tqdm(metadata.items()):
            print(video_name)
            video_path = os.path.join(DATA_ROOT, folder_name + '/' + video_name)
            
            # # Show original video
            # capture = cv2.VideoCapture(video_path)
            # ret, frame = capture.read()
            # cv2.imshow('test', frame)
            # cv2.waitKey(1000)

            frame_indices, frames = get_frames(video_path, 2)
            if len(frames) == 0:
                print('%s: Failed to get images from video' % (frames))
                continue

            results = {}
            for frame_index, frame in zip(frame_indices, frames):
                detection_results = detector.detect_faces(frame)
                if detection_results is None:
                    continue
                
                results[frame_index] = detection_results
                
                for face_index, detection_result in enumerate(detection_results):
                    box = np.array(detection_result['box'])
                    box = box.astype(int)

                    face = frame[box[1]:box[3], box[0]:box[2]]
                    if face.size == 0:
                        continue

                    path = os.path.join(output_folder, '%s-%d-%d.png' % (video_name[:-4], frame_index, face_index))
                    cv2.imwrite(path, cv2.resize(face, (image_size, image_size)))

            attributes['face'] = results
            break

        with open(os.path.join(output_folder, 'metadata.json'), "w") as fp:
            json.dump(metadata, fp)