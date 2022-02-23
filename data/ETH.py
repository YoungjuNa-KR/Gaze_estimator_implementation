import os
import PIL
import torch
import pickle
import numpy as np
from glob import glob
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor

class ETH(torch.utils.data.Dataset):
    def __init__(self, dir_name, transforms=pil_to_tensor):
        # dataset 디렉토리를 기반으로 parse.data_train, test에 따라서
        # 각각 다른 디렉토리에 접근할 수 있도록 한다.
        self.root_dir = os.path.join("./dataset", dir_name)
        self.imgs = os.listdir(self.root_dir)
        # PIL로 이미지를 불러와 텐서로 변환하기 위해 디폴트로 설정한다.
        self.transform = transforms
        # 데이터셋의 개별 이미지의 경로가 저장된다.
        self.data = []

        '''
        label을 로드하는 데에 시간이 많이 소요되기 때문에 딕셔너리 형태로 사전에 정의하고
        이를 학습 과정에서 사용하여 label 검색 시간을 단축시킴.
        기존에는 파일명을 정렬한 후에, 이진탐색 형식으로 사용하였지만,
        key | value 를 갖는 hashmap 사용하여 라벨 호출시간이 단축 및 간편화.
        '''
        
        with open("./dataset/eth_label_dict.pickle", "rb") as f:
            label_dict = pickle.load(f)
        self.label = label_dict
        

        # 개별적으로 .jpg의 확장자(suffix)를 가진 이미지에 접근하여
        # 이미지와 라벨을 저장한다.
        for i, img in enumerate(self.imgs):
            img_path = os.path.join(self.root_dir, img)
            self.data.append(img_path)

    # 클래스 변수로 저장된 이미지와 라벨에 대한 정보를 위한 함수이다.
    def __getitem__(self, idx):
        img_path = self.data[idx]
        img_name = img_path.split('\\')[-1]
        img_name = img_name.split('.')[0]
        label = self.label[img_name]
        # print(label)
        # print(label.size())

        # os.path.basename으로 단일 이미지명을 얻을 수 있도록 한다.
        img_name = os.path.basename(img_path)
        img = PIL.Image.open(img_path)
        img = self.transform(img)

        sample = {"image" : img, "label" : label, "name" : img_name}

        return sample

    def __len__(self):
        return len(self.data)


    
