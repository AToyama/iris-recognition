import os
os.mkdir(f'images_tratadas')

for i in range(60):
    os.mkdir(f'images_tratadas/{i:04d}')
