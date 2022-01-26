import os
import PIL
import torch

from glob import glob
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor

class MPII(torch.utils.data.Dataset):
    def __init__(self, dir_name, transforms=pil_to_tensor):
        # dataset 디렉토리를 기반으로 parse.data_train, test에 따라서
        # 각각 다른 디렉토리에 접근할 수 있도록 한다.
        self.root_dir = os.path.join("./dataset", dir_name)
        self.imgs = os.listdir(self.root_dir)
        # PIL로 이미지를 불러와 텐서로 변환하기 위해 디폴트로 설정한다.
        self.transform = transforms
        # 데이터셋의 개별 이미지의 경로가 저장된다.
        self.data = []
        # 저장된 이미지 경로의 인덱스를 나타낸다. 
        self.label = []
        
        if dir_name == "MPII_train":
            self.latent = "Latent_train"
        elif dir_name == "MPII_validation":
            self.latent = "Latent_validation"

        # 개별적으로 .jpg의 확장자(suffix)를 가진 이미지에 접근하여
        # 이미지와 라벨을 저장한다.
        for i, img in enumerate(self.imgs):
            img_path = os.path.join(self.root_dir, img)
            self.data.append(img_path)
            self.label.append(i)

    # 클래스 변수로 저장된 이미지와 라벨에 대한 정보를 위한 함수이다.
    def __getitem__(self, idx):
        img_path, label = self.data[idx], self.label[idx]
        # os.path.basename으로 단일 이미지명을 얻을 수 있도록 한다.
        img_name = os.path.basename(img_path)
        img = PIL.Image.open(img_path)
        img = self.transform(img)

        latent_name = os.path.splitext(img_name)[0] + ".pt"
        latent_path = os.path.join("./dataset", self.latent, latent_name)
        latent = torch.load(latent_path)
        latent = latent.type('torch.FloatTensor')

        sample = {"image" : img, "label" : label, "name" : img_name, "latent" : latent}

        return sample

    def __len__(self):
        return len(self.data)


    
