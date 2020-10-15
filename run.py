import G6_iris_recognition
import cv2 as cv
import images
from iris import Iris


# inicializa a classe com os métodos para tratar a imagem
img_treater = Iris()

for i in range(60):
    for j in range(20):

        # carrega a imagem
        img = cv.cvtColor(cv.imread(f'images/{i:04d}/{i:04d}_{j:03d}.bmp'), cv.COLOR_BGR2GRAY)

        # tratar a imagem
        imgeq = img_treater.equalize(img)
        
        # gravar a imagem tratada
        cv.imwrite(f'images_tratadas/{i:04d}/{i:04d}_{j:03d}.bmp', imgeq)

