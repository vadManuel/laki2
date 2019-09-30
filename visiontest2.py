import cv2,time,numpy as np

cv2.namedWindow('preview')
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
	rval, frame = vc.read()
else:
	rval = False

fps = 10
t0 = 0

while rval:
	t = time.time() - t0
	if t > 1/fps:
		t0 = time.time()
		# gb = my_gaussian(frame[:,:,0], 3)
		# gg = my_gaussian(frame[:,:,1], 3)
		# gr = my_gaussian(frame[:,:,2], 3)
		# gass = list([gb,gg,gr])

		# gass = np.array(gass, dtype=int)
		# print(np.shape(gass), type(gass))

		# frame = cv2.line(frame,(0,0),(511,511),(255,0,0),5)

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (5, 5), 10)

		gray = np.float32(gray)
		dst = cv2.cornerHarris(gray, 2, 3, 0.04)

		dst = cv2.dilate(dst, None)

		frame[dst > 0.01 * dst.min()] = [0, 0, 0]
		frame[dst < 0.01 * dst.min()] = [0, 255, 0]
		# frame = np.where(frame[dst > 0.01*dst.min()], [0,0,0], [0,255,0])
		# frame = np.where(frame != [0,0,0], [0,255,0], [0,0,0])

		# cv2.imshow('preview', gass)
		cv2.imshow('preview', frame)
	rval, frame = vc.read()
	key = cv2.waitKey(20)
	if key == 27:  # exit on ESC
		break

cv2.destroyWindow('preview')
