import cv2
import numpy as np
import time


def g(x, y, sigma):
	return (1/(2*np.pi*sigma**2))*np.e**(-1*(x**2+y**2)/(2*sigma**2))


def apply_kernel(input, size, kernel, div=1):
	k = size//2
	step = []
	for i in range(size):
		yroll = np.copy(input)
		yroll = np.roll(yroll, i-k, axis=0)
		for j in range(size):
			xroll = np.copy(yroll)
			xroll = np.roll(xroll, j-k, axis=1)*kernel[i, j]
			step.append(xroll)

	step = np.array(step)
	stepsum = np.sum(step, axis=0) / div
	return stepsum


def my_gaussian(input, sigma):
	input = np.array(input, dtype=np.float64)
	size = 6*sigma+1
	k = size//2
	kernel = [[0]*size for _ in range(size)]
	for y in range(-1*k, k+1):
		for x in range(-1*k, k+1):
			kernel[x+k][y+k] = g(x, y, sigma)
	kernel = np.array(kernel, dtype=np.float64)
	return apply_kernel(input, size, kernel, np.sum(kernel))


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
