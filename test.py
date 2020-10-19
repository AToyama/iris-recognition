import cv2 as cv
from res.iris import Iris

img = cv.cvtColor(cv.imread('img_test/0005_012.bmp'), cv.COLOR_BGR2GRAY)

img_treater = Iris()

img = img_treater.equalize(img)
img = img_treater.blur(img)
img = img_treater.sharpening(img)

circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1.2, 100, maxRadius=20)

print(circles)

# img = img_treater.clareia(img, 100, 20)

cv.imwrite('img_test/tratada.bmp', img)