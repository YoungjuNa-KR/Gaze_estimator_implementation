import os
from re import L
from tqdm import tqdm
import shutil

# img split
img_path = 'D:\Gaze_estimator_implementation\dataset\img\inferenced_eth_with_ffhq'
val_path = 'D:\Gaze_estimator_implementation\dataset\img\inferenced_eth_with_ffhq_validation'

files = os.listdir(img_path)
count = 0
for file in tqdm(files):
    # print(os.path.join(data_path, file))
    if count % 10 == 0: 
        shutil.move(os.path.join(img_path, file), os.path.join(val_path, file))
    count += 1


# latent split
latent_path = 'D:\Gaze_estimator_implementation\dataset\latent\ETH_ffhq_latent'
val_path = 'D:\Gaze_estimator_implementation\dataset\latent\ETH_ffhq_latent_validation'

files = os.listdir(latent_path)
count = 0
for file in tqdm(files):
    # print(os.path.join(data_path, file))
    if count % 10 == 0: 
        shutil.move(os.path.join(latent_path, file), os.path.join(val_path, file))
    count += 1