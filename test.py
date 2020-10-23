import cv2 as cv
from res.iris import Iris
import numpy as np

img = cv.cvtColor(cv.imread('img_test/0005_012.bmp'), cv.COLOR_BGR2GRAY)
template = cv.cvtColor(cv.imread('template.bmp'), cv.COLOR_BGR2GRAY)

img1 = img.copy()
img2 = img.copy()

img_treater = Iris()

img2, detected2 = img_treater.cropROI(img, template)
img1, detected1 = img_treater.detectROI(img, template)


cv.imwrite('img_test/tratada_detected.bmp', img1)
cv.imwrite('img_test/tratada_croped.bmp', img2)