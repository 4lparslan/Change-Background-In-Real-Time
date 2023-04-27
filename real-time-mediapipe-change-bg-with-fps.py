import numpy as np
import mediapipe as mp
import cv2
import time

segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection = 1)

prev_frame_time = 0
new_frame_time = 0

cap = cv2.VideoCapture(0)
#cap.set(3,1280)
#cap.set(4,720)

while cap.isOpened():
    ret, frame = cap.read()
    height, width, channel = frame.shape
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(round(fps))

    results = segmentation.process(RGB)
    mask = results.segmentation_mask

    rsm = np.stack((mask,)*3, axis=-1)
    condition = rsm > 0.6
    condition = np.reshape(condition, (height,width,3))

    blur_frame = cv2.blur(frame, ksize=(7, 7))

    output = np.where(condition,frame,blur_frame)

    merged = np.concatenate((output, frame), axis=1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(merged, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("output",merged)

    k = cv2.waitKey(30)

    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
