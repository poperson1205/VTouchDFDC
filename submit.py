# MODE = 'KAGGLE'
MODE = 'LOCAL'

if MODE == 'LOCAL':
    model_parameter_path = './binary_classifier.pth'
    test_dataset_dir = '/media/vtouchinc02/database/RawData/deepfake/dfdc_train_part_48'
    output_submission_file_path = './submission.csv'
    metadata_path = test_dataset_dir + '/metadata.json'
elif MODE == 'KAGGLE':
    model_parameter_path = '/kaggle/input/models/binary_classifier.pth'
    test_dataset_dir = '/kaggle/input/deepfake-detection-challenge/test_videos'
    output_submission_file_path = '/kaggle/working/submission.csv'
    efficientnet_pakage_path = '../input/efficientnetpytorch/EfficientNet-PyTorch'
    sys.path.append(efficientnet_pakage_path)
    os.system('pip install /kaggle/input/mtcnnwheel/mtcnn-0.1.0-py3-none-any.whl')

import os, sys, random
import numpy as np
import pandas as pd
import cv2
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

from tqdm import tqdm

import matplotlib.pyplot as plt

from efficientnet_pytorch import EfficientNet

import retina_face_detector



image_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean,std)
gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
detector = retina_face_detector.RetinaFaceDetector(
    network='mobile0.25',
    trained_model='weights/mobilenet0.25_Final.pth',
    vis_thres=0.9,
    show_image=False)



def get_frames(video_path):
    capture = cv2.VideoCapture(video_path)

    num_frames = 17
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



def predict(video_path):
    default_prediction = 0.5

    # Get first frame
    frames = get_frames(video_path)
    if len(frames) == 0:
        return default_prediction

    list_detection_results = detector.detect_faces_batch(frames)

    faces = []
    for frame, detection_results in zip(frames, list_detection_results):
        if detection_results is None:
            continue
        
        for detection_result in detection_results:
            box = np.array(detection_result['box']).astype(int)
            face = frame[box[1]:box[3], box[0]:box[2]]
            if face.size == 0:
                continue

            # Convert image
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (image_size, image_size))
            faces.append(face)

    if len(faces) == 0:
        return default_prediction

    x = np.stack(faces)
    x = torch.tensor(x, device=gpu).float().permute((0, 3, 1, 2))
    for i in range(len(x)):
        x[i] = normalize_transform(x[i] / 255.)

    with torch.no_grad():
        y = torch.sigmoid(model(x).squeeze()).mean().item()

    return y



model = EfficientNet.from_name('efficientnet-b0', {'num_classes': 1})
checkpoint = torch.load(model_parameter_path, map_location=gpu)
model.load_state_dict(checkpoint)
model.cuda()
model.train(False)

predictions = []

if MODE == 'KAGGLE':
    for root, _, filenames in os.walk(test_dataset_dir):
        for count, filename in enumerate(filenames, 1):
            if filename.endswith('.mp4'):
                filepath = os.path.join(root, filename)
                prediction = predict(filepath)
                predictions.append([filename, prediction])
                print('%s: %f (%3d / %3d)' % (filename, prediction, count, len(filenames)))
    df = pd.DataFrame(data=predictions, columns=['filename', 'label'])
    df.sort_values('filename').to_csv(output_submission_file_path, index=False)

elif MODE == 'LOCAL':
    with open(metadata_path, 'r') as fp:
        metadata = json.load(fp)

    num_video = len(metadata)

    real_videos = []
    fake_videos = []
    for filename, attributes in metadata.items():
        if attributes['label'] == 'REAL':
            real_videos.append(filename)
        else:
            fake_videos.append(filename)

    real_videos_sampled = random.sample(real_videos, 100)
    fake_videos_sampled = random.sample(fake_videos, 100)

    for i in tqdm(range(100)):
        filename = real_videos_sampled[i]
        filepath = os.path.join(test_dataset_dir, filename)
        prediction = predict(filepath)
        predictions.append([filename, prediction])
        print('%s (REAL): %f' % (filename, prediction))

        filename = fake_videos_sampled[i]
        filepath = os.path.join(test_dataset_dir, filename)
        prediction = predict(filepath)
        predictions.append([filename, prediction])
        print('%s (FAKE): %f' % (filename, prediction))
    df = pd.DataFrame(data=predictions, columns=['filename', 'label'])
    df.to_csv(output_submission_file_path, index=False)

print('complete!')