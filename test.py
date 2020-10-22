import cv2 as cv
from res.iris import Iris
import numpy as np

img = cv.cvtColor(cv.imread('img_test/0005_012.bmp'), cv.COLOR_BGR2GRAY)
template = cv.cvtColor(cv.imread('img_test/template.bmp'), cv.COLOR_BGR2GRAY)

img_treater = Iris()

# Metodo 1

# img = img_treater.equalize(img)
# img = img_treater.blur(img)
# img = img_treater.sharpening(img)
# img = img_treater.clareia(img, 100, 20)


img = img_treater.detectROI(img, template, 0.7)


cv.imwrite('img_test/tratada.bmp', img)