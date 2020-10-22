import os
os.system("mkdir images_tratadas")

for i in range(60):
    os.system(f'mkdir images_tratadas/{i:04d}')
