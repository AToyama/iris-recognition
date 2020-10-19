import cv2 as cv
import numpy as np

class Iris:
    def __init__(self):
        pass

    def equalize(self, img):
        return cv.equalizeHist(img)

    def blur(self, img):
        return cv.GaussianBlur(img, (5, 5), 0)

    def sharpening(self, img):
        sharp = np.array([
            [ 0,-1, 0],
            [-1, 5,-1],
            [ 0,-1, 0]
        ])
      
        return cv.filter2D(img,-1,sharp) 

    def clareia(self, img, T, M):
        
        imgf = img.copy()

        row, col = img.shape

        for i in range(row):
            for j in range(col):
                if img[i, j] < T:
                    imgf[i, j] += M

        return imgf
 
