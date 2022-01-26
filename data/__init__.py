import os
from data.Latent import Latent
from data.MPII import MPII
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import DataLoader

# from importlib import import_module


class Data:
    def __init__(self, args):

        self.loader_train = None
        if not args.test_only:
            
            # 현재는 필요하지 않음. (공부용)
            # HR / Bicubic 필요 없기 때문에 사용하지 않는다.

            # importlib.import_module("from $$ import $$") 형식으로 생각하면 된다.
            # 모듈을 처음부터 import 하지 않고 로직에 따라 (모듈 명을 변수로 받아) 이것을 이용해 모듈을 import 한다.
            """
            module_train = import_module('data.' + args.data_train.lower())
            모듈을 상황에 따라 유동적으로 사용하기 위해 import_module() 함수를 사용한다.
            """
            # getattr(object, attribute)
            # 지정한 object의 속성을 문자열 형태로 접근이 가능하도록 하는 함수이다.
            # 이를 통해 클래스 변수에 이용하여 접근이 가능하다.
            """
            trainset = getattr(module_train, args.data_train)(args)
            불러온 모듈에서 대응하는 문자열을 통해 trainset을 불러오기 위해 사용한다.
            """

            # DataLoader 부분을 수정하였다. 
            # torchvision.datasets.ImageFolder 함수를 통해 경로로 파일을 가져오고, 그 과정에서 PIL 파일을 tensor로 변환한다.

            # 이러한 로드의 단점은 불러온 파일명을 직접적으로 확인할 수 없다는 것이다.
            # 하지만 trainer 파일에서 datasets의 __getitem__ 클래스 함수의 속성값을 이용하여 배치 단위의 파일명을 불러오도록
            # 코드를 작성하여 수정하였다.

            if args.data_train == "MPII_train":
                trainset = MPII(args.data_train)
            elif args.data_train == "Latent_train":
                trainset = Latent(args.data_train)
            else:
                print("데이터셋을 확인하세요 : MPII_train/Latent_train")

            self.loader_train = DataLoader(
                trainset,
                batch_size=args.batch_size,
                num_workers=args.n_threads,
                shuffle=True,
                pin_memory=not args.cpu
            )

        # trainset과 같은 맥락에서 사용된다. (공부용)
        """
        module_test = import_module('data.' +args.data_test.lower())
        testset = getattr(module_test, args.data_test)(args, name=args.data_test ,train=False)
        """

        if args.data_test == "MPII_validation":
                testset = MPII(args.data_test)
        elif args.data_test == "Latent_validation":
                testset = Latent(args.data_test)
        else:
            print("데이터셋을 확인하세요 : MPII/Latent")

        
        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            num_workers=1,
            shuffle=True,
            pin_memory=not args.cpu
        )

