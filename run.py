import G6_iris_recognition
import cv2 as cv
import images

from res.iris import Iris

# inicializa a classe com os métodos para tratar a imagem
img_treater = Iris()
template = cv.cvtColor(cv.imread('template.png'), cv.COLOR_BGR2GRAY)

for i in range(60):
    for j in range(20):

        # carrega a imagem
        img = cv.cvtColor(cv.imread(f'images/{i:04d}/{i:04d}_{j:03d}.bmp'), cv.COLOR_BGR2GRAY)

        # teste de detecção de iris
        img, detectou = img_treater.cropROI(img, template)

        img = img_treater.blur(img)
        img = img_treater.equalize(img)
        img = img_treater.sharpening(img)

        
        # gravar a imagem tratada
        if detectou:
            print(f'[\033[92mSUCCESS\033[0m] {i:04d}/{i:04d}_{j:03d}.bmp detected!')
            cv.imwrite(f'images_tratadas/{i:04d}/{i:04d}_{j:03d}.bmp', img)
        else:
            print(f'[\033[91mFAIL\033[0m] Image {i:04d}/{i:04d}_{j:03d}.bmp not detected')



# G6_iris_recognition.iris_model_train("images","res/model.pickle")
# iris_name= G6_iris_recognition.iris_model_test("res/model.pickle","images/0000/0000_000.bmp")
# print(iris_name)