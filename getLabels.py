from time import sleep
import subprocess
import os

# Parameters
img_folder = 'temp'

# Load filenames of all .png files
images = [img for img in os.listdir(img_folder) if '.png' in img]

# Isolate unique image labels
labels = []
for file_name in images:
    frame_label = file_name.split('-')[0]
    if frame_label not in labels:
        print("frame_label: ", frame_label)
        labels.append(frame_label)

print("Labels: ", labels)

for label in labels:
    print('Running Subprocess: python buildVideo.py ', label)
    os.system('python buildVideo.py ' + label)
    sleep(1)