import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

while True:
	ret, frame = video_capture.read()
	frame = cv2.flip(frame, 1)
	
	blur_frame = cv2.blur(frame,ksize = (7,7))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	for (x,y,w,h) in faces:
		face = frame[y:y+h,x:x+w]
		blur_frame[y:y+h,x:x+w] = face
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

	merged = np.concatenate((blur_frame,frame), axis=1)
	
	cv2.imshow("Video", merged)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_capture.release()
cv2.destroyAllWindows()