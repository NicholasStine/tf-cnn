# Imports
import numpy as np
import cv2
import sys
import os

# Parameters
img_folder = 'temp'
vid_name = 'sonogram.avi'
label = str(list(sys.argv)[1])
print("Label: ", label)

# Load filenames of all .png files
images = [img for img in os.listdir(img_folder) if '.png' in img and label in img]

# Get Frame Shape
frame = cv2.imread(os.path.join(img_folder, images[0]))
h, w, layers = frame.shape

# Initialize the video
video = cv2.VideoWriter("output_videos/" + label + ".avi", cv2.VideoWriter_fourcc(*'XVID'), 7, (w, h))
frame_list = []

# Build an array of dictionaries with the loaded cv2 image and it's HSV value for V (brightness)
for image in [img for img in images if label in img]:
    # print(image)
    hsv_f = cv2.cvtColor(cv2.imread(os.path.join(img_folder, image)), cv2.COLOR_BGR2HSV)
    frame_list.append({
        0: cv2.imread(os.path.join(img_folder, image)),
        1: int(1 / ((hsv_f[0][0][1] + hsv_f[0][0][2]) / 2))
})

# Sort the frames objects by their hsv value of choice
sorted_frames = sorted(frame_list, key=lambda item: item[1])
print(sorted_frames)

# Write the frames to the video
for frame in sorted_frames:
    video.write(frame[0])

# Wait for the video object
video.release()
cv2.destroyAllWindows()

