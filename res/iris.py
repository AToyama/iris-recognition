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
    
    def detectROI(self, img, template):

        imgD = img.copy()

        #abre o template e o converte para níveis de cinza
        template = cv.GaussianBlur(template, (7, 7), 0)

        #comprimento e largura do template
        width,height=template.shape

        #template matching, que devolve o nível de acurácia
        match = cv.matchTemplate(imgD, template, cv.TM_CCOEFF_NORMED)

        #obtém as posições onde o template gerou níveis de acurácia maiores que um limiar
        threshold = 0.6
        position = np.where(match >= threshold)
        last_pos = (0,0)
        iris = False

        #desenha os retângulos com as regiões encontradas
        for point in zip(*position[::-1]): 
            if abs(point[0] - last_pos[0]) < 20 or abs(point[1] - last_pos[1]) < 20:
                continue
        
            cv.rectangle(imgD, (point[0]-60, point[1]-40), (point[0] + width + 60, point[1] + height + 60), (0, 204, 153), 5)
            last_pos = point
            iris = True


        return imgD, iris

    def cropROI(self, img, template):

        imgC = img.copy()

        #abre o template e o converte para níveis de cinza
        template = cv.GaussianBlur(template, (7, 7), 0)

        #comprimento e largura do template
        width,height=template.shape

        #template matching, que devolve o nível de acurácia
        match = cv.matchTemplate(imgC, template, cv.TM_CCOEFF_NORMED)

        #obtém as posições onde o template gerou níveis de acurácia maiores que um limiar
        threshold = 0.6
        position = np.where(match >= threshold)
        last_pos = (0,0)
        iris = False

        #desenha os retângulos com as regiões encontradas
        for point in zip(*position[::-1]):
            if abs(point[0] - last_pos[0]) < 20 or abs(point[1] - last_pos[1]) < 20:
                continue
            
            y1 = point[1] - 40
            x1 = point[0] - 80
            y2 = point[1] + height + 80
            x2 = point[0] + width + 80

            # cv.rectangle(imgC, (point[0]-70, point[1]-70), (point[0] + width + 70, point[1] + height + 70), (0, 204, 153), 5)
            imgC = imgC[y1:y2, x1:x2]
            last_pos = point
            iris = True
            break


        return imgC, iris
 
