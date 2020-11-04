import G6_iris_recognition

def test_model():

    right = 0
    wrong = 0

    real = []
    predict = []

    for i in range(60):
        for j in range(20):
            
            print(f'imagem testada = {i:04d}/{i:04d}_{j:03d}.bmp')
            iris_name = G6_iris_recognition.iris_model_test("res/model.pickle",f"images_tratadas/{i:04d}/{i:04d}_{j:03d}.bmp")
            print(iris_name)

            if iris_name == f'{i:04d}':
                right += 1
            else:
                wrong += 1

            real.append(f'{i:04d}')
            predict.append(iris_name)

    print('\033[92m    END OF TEST\033[0m')
    print(f'{right} rights - {wrong} wrongs')
    print('\033[92m    Matriz Confusão\033[0m')
    print([real, predict])

test_model()