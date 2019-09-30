'''
Testing edge detection
'''


import numpy as np
import cv2
from matplotlib import pyplot as plt


def main():
    '''
    main
    '''

    img = 'image10.jpg'
    # img = 'red_square_blue_B.png'
    img_0 = cv2.imread(img)
    img_1 = cv2.imread(img)

    gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 7)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 1, 3, 0.04)

    img_1[dst > 0.01 * dst.min()] = [0, 0, 0]
    img_1[dst <= 0.01 * dst.min()] = [255, 255, 255]

    plt.subplot(1, 2, 1)
    plt.imshow(img_0)
    plt.title('Original')

    plt.subplot(1, 2, 2)
    plt.imshow(img_1)
    plt.title('Altered')

    plt.show()

    return None


if __name__ == '__main__':
    main()
