# https://prince-canuma.medium.com/image-pre-processing-c1aec0be3edf,
# https://www.analyticsvidhya.com/blog/2021/05/30-useful-methods-from-python-os-module/
import cv2
import os
import numpy as np
from progressbar import ProgressBar
from parameters import *


def extract_frames(video_path):
    # https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
    frames = []
    video = cv2.VideoCapture(video_path)
    count_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = max(int(count_frames / SEQUENCE_LENGTH), 1) # Using every frame would be too much for the computers :(

    for i in range(SEQUENCE_LENGTH):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * skip_frames)
        success, frame = video.read()
        if not success:
            break
        resized = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # http://dev.ipol.im/~nmonzon/Normalization.pdf
        normalized_frame = resized / 255
        frames.append(normalized_frame)

    video.release()
    return frames


print("building dataset")
def build_dataset(): #given classes we can create a dataset
    features = []
    labels = []
    path2videos = []

    for index, name in enumerate(CLASSES_LIST):
        print(f"Extracting frames of {name}")
        files_list = os.listdir(os.path.join(DIR, name))

        for file_name in files_list:
            video_file_path = os.path.join(DIR, name, file_name)
            frames = extract_frames(video_file_path)

            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(index)
                path2videos.append(video_file_path)
    
    # list2array
    features = np.asarray(features)
    labels = np.array(labels)
    
    return features, labels, path2videos


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





