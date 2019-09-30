'''
Testing edge detection
'''


import time
import numpy as np
import cv2


def main():
    '''
    main
    '''

    cv2.namedWindow('preview')
    video = cv2.VideoCapture(0)

    if video.isOpened():
        rval, frame = video.read()
    else:
        rval = False

    fps = 10
    time_0 = 0

    while rval:
        time_1 = time.time() - time_0
        if time_1 > 1/fps:
            time_0 = time.time()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 10)

            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray, 2, 3, 0.04)

            dst = cv2.dilate(dst, None)

            frame[dst > 0.01 * dst.min()] = [0, 0, 0]
            frame[dst < 0.01 * dst.min()] = [255, 255, 255]
            cv2.imshow('preview', frame)

        rval, frame = video.read()

        key = cv2.waitKey(20) & 0xff
        if key == 27 or key == ord('q'):  # exit on ESC
            break

    cv2.destroyWindow('preview')

    return None


if __name__ == '__main__':
    main()
