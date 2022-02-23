import os
import cv2
from tqdm import tqdm
from glob import glob


dir_path = "D:\Gaze_estimator_implementation\dataset\img\inferenced_eth_with_ffhq"
# imgs = glob(dir_path+'/*')
imgs = os.listdir(dir_path)

for img_name in tqdm(imgs):
    img = cv2.imread(os.path.join(dir_path, img_name))
    if img.shape[0] != 256:
        img_resized = cv2.resize(img, (256,256))
        cv2.imwrite(os.path.join(dir_path, img_name), img_resized)