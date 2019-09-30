'''
Testing edge detection
'''


import numpy as np
import cv2
from matplotlib import pyplot as plt


def do_the_thing(img):
    '''
    does the thing
    '''
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(
        centroids), (5, 5), (-1, -1), criteria)

    res = np.hstack((centroids, corners))
    res = np.int0(res)
    img[res[:, 1], res[:, 0]] = [0, 0, 255]
    img[res[:, 3], res[:, 2]] = [0, 255, 0]

    for i in range(len(res)):
        for j in range(i+1,len(res)):
            img = cv2.line(img, (res[i, 3], res[i, 2]), (res[j, 3], res[j, 2]), (255, 0, 0), 2)

    return img


def main():
    '''
    main
    '''

    img = ['image10.jpg', 'red_square_blue_B.png', 'purple_square_white_A.jpg', 'image12.png']
    res = [do_the_thing(i) for i in img]

    for i in range(len(img)):
        plt.subplot(2,2,i+1)
        plt.title(img[i])
        plt.imshow(res[i])
    
    plt.show()

    return None


if __name__ == '__main__':
    main()
