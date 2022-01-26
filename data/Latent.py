import os
import PIL
import torch

from glob import glob
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor

class Latent(torch.utils.data.Dataset):
    def __init__(self, dir_name, transforms=None):
        # dataset 디렉토리를 기반으로 parse.data_train, test에 따라서
        # 각각 다른 디렉토리에 접근할 수 있도록 한다.
        self.root_dir = os.path.join("./dataset", dir_name)
        self.imgs = os.listdir(self.root_dir)
        self.transform = None
        # 데이터셋의 개별 텐서의 경로가 저장된다.
        self.data = []
        # 저장된 텐서 경로의 인덱스를 나타낸다. 
        self.label = []
        
        # 개별적으로 텐서에 접근하고, 대응하는 라벨을 저장한다.
        for i, img in enumerate(self.imgs):
            img_path = os.path.join(self.root_dir, img)
            for img in glob(os.path.join(img_path)):
                self.data.append(img)
                self.label.append(i)
    
    # 클래스 변수로 저장된 이미지와 라벨에 대한 정보를 위한 함수이다.
    def __getitem__(self, idx):
        img_path, label = self.data[idx], self.label[idx]
        # os.path.basename으로 단일 이미지명을 얻을 수 있도록 한다.
        img_name = os.path.basename(img_path)
        img = torch.load(img_path)
        img = img.type('torch.FloatTensor')

        sample = {"image" : img, "label" : label, "name" : img_name}

        return sample

    def __len__(self):
        return len(self.data)


    
