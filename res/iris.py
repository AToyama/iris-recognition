import cv2 as cv

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
     
 
