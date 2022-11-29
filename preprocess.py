# Testing code on smaller examples and reduced parameters to save on time
# Increaseing height, width, or sequence length should bring better resuts
# https://prince-canuma.medium.com/image-pre-processing-c1aec0be3edf, https://www.analyticsvidhya.com/blog/2021/05/30-useful-methods-from-python-os-module/
import cv2
import os
import numpy as np

# parameters
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
DIR = "UCF50/UCF50"
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]

def extract_frames(video_path):
    # https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
    frames = []
    video_render = cv2.VideoCapture(video_path)
    count_frames = int(video_render.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = max(int(count_frames / SEQUENCE_LENGTH), 1) # Using every frame would be too much for my pc :(

    for counter in range(SEQUENCE_LENGTH):
        video_render.set(cv2.CAP_PROP_POS_FRAMES, counter * skip_frames)
        success, frame = video_render.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # http://dev.ipol.im/~nmonzon/Normalization.pdf
        normalized_frame = resized_frame / 255
        frames.append(normalized_frame)

    video_render.release()
    return frames

def build_dataset():#given classes we can create a dataset

    features = []
    labels = []
    video_files_path = []

    for i, name in enumerate(CLASSES_LIST):
        print(f"Extracting data of class {name}")
        files = os.listdir(os.path.join(DIR, name))

        for file_name in files:
            video_path = os.path.join(DIR, name, file_name)
            frames = extract_frames(video_path)

            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(i)
                video_files_path(video_path)
    
    # list2array
    features = np.asarray(features)
    labels = np.array(labels)

    return features, labels, video_files_path


def to_categorical(y, num_classes=None, dtype="float32"):
    # https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py#L9-L37
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical





