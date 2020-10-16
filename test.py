import cv2 as cv
from res.iris import Iris

img = cv.cvtColor(cv.imread('img_test/0005_012.bmp'), cv.COLOR_BGR2GRAY)

img_treater = Iris()

img_t = img_treater.blur(img)
img_t = img_treater.equalize(img_t)

cv.imwrite('img_test/tratada.bmp', img_t)