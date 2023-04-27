import numpy as np
import mediapipe as mp
import cv2

segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection = 1)

cap = cv2.VideoCapture(0)
#cap.set(3,1280)
#cap.set(4,720)

while cap.isOpened():
    ret, frame = cap.read()
    height, width, channel = frame.shape
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = segmentation.process(RGB)
    mask = results.segmentation_mask
    cv2.imshow("mask", mask)
    rsm = np.stack((mask,)*3, axis=-1)
    condition = rsm > 0.4
    condition = np.reshape(condition, (height,width,3))

    blur_frame = cv2.blur(frame, ksize=(15,15))

    output = np.where(condition,frame,blur_frame)

    merged = np.concatenate((output, frame), axis=1)
    cv2.imshow("output",merged)

    k = cv2.waitKey(30)

    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
