import cv2

cap = cv2.VideoCapture("videos/IMG-4604.mov")
has_next_frame, frame = cap.read()
n = 0

while has_next_frame:
    print("Writing frame: ", n)
    cv2.imwrite("raw_frames/frame%d.jpg" % n, frame)
    has_next_frame, frame = cap.read()
    n += 1
