# figure for presentation purposes
# first frame of a video from each category

import matplotlib.pyplot as plt
import os
import random
import cv2

plt.figure(figsize=(20, 20))
all_classes = os.listdir('UCF50/UCF50')
#print(all_classes)

random_range = random.sample(range(len(all_classes)), 20)

for counter, random_index in enumerate(random_range, 1):
    random_class_name = all_classes[random_index]
    video_names = os.listdir(f'UCF50/UCF50/{random_class_name}')
    random_video = random.choice(video_names)
    render_video = cv2.VideoCapture(f'UCF50/UCF50/{random_class_name}/{random_video}')
    _, bgr_frame = render_video.read()
    render_video.release()
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    cv2.putText(rgb_frame, random_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    plt.subplot(5, 4, counter);plt.imshow(rgb_frame);plt.axis('off')
    plt.savefig('datasetpreview.png')

