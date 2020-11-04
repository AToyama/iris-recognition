import G6_iris_recognition
import cv2 as cv
import images
import model_test

from res.iris import Iris

# inicializa a classe com os métodos para tratar a imagem
img_treater = Iris()
template = cv.cvtColor(cv.imread('template2.png'), cv.COLOR_BGR2GRAY)

for i in range(60):
    for j in range(20):

        img_path = f'{i:04d}/{i:04d}_{j:03d}.bmp'

        # carrega a imagem
        img = cv.cvtColor(cv.imread(f'images/{img_path}'), cv.COLOR_BGR2GRAY)

        # teste de detecção de iris
        # img, detectou, rects = img_treater.detectROI(img, template)
        img, detectou = img_treater.cropROI(img, template)
        
        # img = img_treater.blur(img)
        img = img_treater.equalize(img)
        img = img_treater.sharpening(img)

        
        # gravar a imagem tratada
        if detectou:
            print(f'\033[92m[SUCCESS]\033[0m {img_path} detected! -> {img.shape}')
            cv.imwrite(f'images_tratadas/{img_path}', img)
        else:
            print(f'\033[91m[FAIL]\033[0m Image {img_path} not detected -> {img.shape}')

# Treinando o modelo
G6_iris_recognition.iris_model_train("images_tratadas","res/model.pickle")

# Testando o modelo
model_test.test_model()
