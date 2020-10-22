import cv2 as cv
from res.iris import Iris
import numpy as np

img = cv.cvtColor(cv.imread('img_test/0050_012.bmp'), cv.COLOR_BGR2GRAY)
template = cv.cvtColor(cv.imread('template.bmp'), cv.COLOR_BGR2GRAY)

img_treater = Iris()

img, detected = img_treater.cropROI(img, template)
# img, detected = img_treater.detectROI(img, template, 0.7)


cv.imwrite('img_test/tratada.bmp', img)