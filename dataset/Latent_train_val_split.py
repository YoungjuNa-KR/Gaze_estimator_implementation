import os
from tqdm import tqdm

train_path = "./dataset/MPII_train"
train_img_list = os.listdir(train_path)

validation_path = "./dataset/MPII_validation"
validation_img_list = os.listdir(validation_path)

pt_path = "./dataset/Latents"
pt_list = os.listdir(pt_path)

train_latent_path = "./dataset/Latent_train"
validation_latent_path = "./dataset/Latent_validation"

for pt in tqdm(pt_list):
    pt_name = os.path.splitext(pt)[0]

    for train_img in train_img_list:
        if pt_name == train_img[:-4]:
            src_path = os.path.join(pt_path, pt)
            dst_path = os.path.join(train_latent_path, pt)
            os.rename(src_path,dst_path)

    for validation_img in validation_img_list:
        if pt_name == validation_img[:-4]:
            src_path = os.path.join(pt_path, pt)
            dst_path = os.path.join(validation_latent_path, pt)
            os.rename(src_path,dst_path)