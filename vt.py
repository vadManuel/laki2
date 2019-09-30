import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv2 import imread, GaussianBlur, threshold, cvtColor
from skimage.feature import hog
from sklearn.preprocessing import normalize
from skimage.color import rgb2gray
from skimage import data, exposure
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def do_hog(image):
    image = imread(image)
    image = rgb2gray(image)
    image = normalize(image)
    image = GaussianBlur(image, (5, 5), 3)

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(3, 3), visualize=True, multichannel=False)

    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 255))
    return hog_image

hog = do_hog('image04.jpg')
thresh = threshold(hog, 0.00001, 255, cv2.THRESH_BINARY)[1]

plt.imshow(thresh)
plt.show()
