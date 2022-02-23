import os
from tqdm import tqdm
import shutil

img_train_path = './dataset/ETH_train'
latent_train_path = './dataset/ETH_latent_train'
img_validation_path = './dataset/ETH_validation'
latent_validation_path = './dataset/ETH_latent_validation'

img_train_list = os.listdir(img_train_path)
latent_train_list = os.listdir(latent_train_path)
img_validation_list = os.listdir(img_validation_path)
latent_validation_list = os.listdir(latent_validation_path)


for e in tqdm(latent_train_list):
    name = e.split('.')[0]
    img_name = name + '.jpg'
    pt_name = name + '.pt'
    if img_name not in img_train_list:
        src = os.path.join(img_validation_path, img_name)
        dst = os.path.join(img_train_path, img_name)
        # print("src:", src)
        # print("dst:", dst)
        shutil.move(src, dst)
        

for e in tqdm(latent_validation_list):
    name = e.split('.')[0]
    img_name = name + '.jpg'
    pt_name = name + '.pt'
    if img_name not in img_validation_list:
        src = os.path.join(img_train_path, img_name)
        dst = os.path.join(img_validation_path, img_name)
        shutil.move(src, dst)