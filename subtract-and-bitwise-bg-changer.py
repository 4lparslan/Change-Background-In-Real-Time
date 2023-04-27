import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

bgImg = cv.imread("green.jpg")

takeBgImage = 0

while(1):
	ret, img = cap.read()

	if img is None:
		break

	rows,cols,channels = img.shape
	roi = bgImg[0:rows, 0:cols]

	if takeBgImage == 0:
		bgReference = img

	# create mask
	diff1 = cv.subtract(img, bgReference)
	diff2 = cv.subtract(bgReference, img)

	diff = diff1 + diff2
	diff[abs(diff) < 25.0] = 0

	cv.imshow("diff",diff)

	gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
	gray[np.abs(gray) < 10] = 0
	fgMask = gray
	cv.imshow("Gray",fgMask)

	#opening to reduce noise
	kernel = np.ones((3,3), np.uint8)

	fgMask = cv.erode(fgMask, kernel, iterations = 2)
	fgMask = cv.dilate(fgMask, kernel, iterations = 2)

	fgMask[fgMask > 5] = 255

	cv.imshow("FG Mask",fgMask)

	#invert mask
	fgMask_inv = cv.bitwise_not(fgMask)
	cv.imshow("FG Mask inv",fgMask_inv)

	bgImage = cv.bitwise_and(roi, roi, mask = fgMask_inv)
	fgImage = cv.bitwise_and(img, img, mask = fgMask)

	final = cv.add(bgImage,fgImage)

	cv.imshow("Background Removed", final)
	cv.imshow("Original", img)

	key = cv.waitKey(1)

	if key == ord("q"): 
		break
	elif key == ord("a"):
		takeBgImage = 1
		print("Background Captured")
	elif key == ord("s"):
		takeBgImage = 0
		print("Reset - Ready For New Capture")

cap.release()
cv.destroyAllWindows()






















