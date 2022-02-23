import os
from re import L
from tqdm import tqdm
import shutil

# img split
img_train_path = 'D:\Gaze_estimator_implementation\dataset\img\inferenced_eth_with_ffhq'
img_val_path = 'D:\Gaze_estimator_implementation\dataset\img\inferenced_eth_with_ffhq_validation'

latent_train_path = 'D:\Gaze_estimator_implementation\dataset\latent\ETH_ffhq_latent'
latent_val_path = 'D:\Gaze_estimator_implementation\dataset\latent\ETH_ffhq_latent_validation'

latent_train_list = os.listdir(latent_train_path)
latent_val_list = os.listdir(latent_val_path)

# count = 0
# files = os.listdir(img_train_path)
# # img trainset
# for file in tqdm(files):
#     # print(os.path.join(data_path, file))
#     # if count % 10 == 0:
#     name = file.split(".")[0]
#     latent = name + '.pt'
#     # latent = os.path.join(latent_train_path, latent)
#     if latent not in latent_train_list:
#         shutil.move(os.path.join(latent_val_path, latent), os.path.join(latent_train_path, latent))
#         count += 1
    
# print(count)

count = 0
files = os.listdir(img_val_path)
# img testset
for file in tqdm(files):
    # print(os.path.join(data_path, file))
    # if count % 10 == 0:
    name = file.split(".")[0]
    latent = name + '.pt'
    
    if latent not in latent_val_list:
        shutil.move(os.path.join(latent_train_path, latent), os.path.join(latent_val_path, latent))
        count += 1
    
print(count)
