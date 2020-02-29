import os
import json
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

from mtcnn import MTCNN

from tqdm import tqdm

DATA_ROOT = '/media/vtouchinc02/database/RawData/deepfake'
OUTPUT_ROOT = '/media/vtouchinc02/database/RawData/deepfake-faces'

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

    return frames

def get_faces(video_path, num_frames=17):
    frames = get_frames(video_path, num_frames)
    if len(frames) == 0:
        return []

    faces = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(frame)
        if results is None or len(results) == 0:
            continue

        # Get face
        box = np.array(results[0]['box'])
        box = box.astype(int)
        face = frame[box[1]:(box[1]+box[3]), box[0]:(box[0]+box[2])]
        if face.size == 0:
            continue

        # Convert image to tensor        
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (image_size, image_size))
        faces.append(face)

    return faces

if __name__ == '__main__':

    image_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize_transform = Normalize(mean,std)
    gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    detector = MTCNN()

    for folder_name in tqdm(os.listdir(DATA_ROOT)):
        print(folder_name)
        metadata_path = os.path.join(DATA_ROOT, folder_name + '/metadata.json')
        with open(metadata_path) as metadata_fp:
            metadata = json.load(metadata_fp)
        for video_name, attributes in tqdm(metadata.items()):
            print(video_name)
            video_path = os.path.join(DATA_ROOT, folder_name + '/' + video_name)
            # capture = cv2.VideoCapture(video_path)
            # ret, frame = capture.read()
            # cv2.imshow('test', frame)
            # cv2.waitKey(1000)
            faces = get_faces(video_path, 31)
            for index, face in zip(range(len(faces)), faces):
                path = os.path.join(OUTPUT_ROOT, folder_name + '/' + video_name[:-4] + '-' + str(index) + '.png')
                cv2.imwrite(path, face)
