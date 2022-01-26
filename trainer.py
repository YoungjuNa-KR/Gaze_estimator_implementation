import torch
import numpy as np


import utility
from decimal import Decimal
from tqdm import tqdm
from option import args

from torchvision import transforms 
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import copy
import h5py
import os
import cv2
import numpy as np
from torchvision.utils import save_image
import torchvision
from getGazeLoss import *
from sklearn.preprocessing import MinMaxScaler


matplotlib.use("Agg")
class Trainer():
    def __init__(self, opt, loader, gaze_model, loss,ckp):
        self.opt = opt
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = gaze_model
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.gaze_model_scheduler = utility.make_gaze_model_scheduler(opt, self.optimizer)
        self.loss = loss
        self.error_last = 1e8
        self.iteration_count = 0
        self.endPoint_flag = True
    
    def train(self):
        # 1 epoch의 평균 시선 추정 손실값과 오류 각도를 연산하기 위해 정의한다. 
        total_gaze_loss = 0
        total_angular_error = 0

        # 학습 상황에 따라 현재 epoch와 learning rate를 파악하기 위해 사용한다. 
        epoch = self.gaze_model_scheduler.last_epoch + 1
        lr = self.gaze_model_scheduler.get_last_lr()[0]
        self.ckp.set_epoch(epoch)

        # 시선 추정 오류를 계산하기 위해 라벨을 불러올 수 있도록 한다.
        label_txt = open("./dataset/MPII_gaze_label.txt" , "r")
        labels = label_txt.readlines()

        # CMD 창에 현재 epoch와 learning rate를 출력하도록 한다.
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()

        """
        TRAIN
        """

        for batch, (samples) in enumerate(self.loader_train):
            self.iteration_count += 1

            # print(samples)
            # Samples라는 배치 단위의 딕셔너리에서 필요한 value 값을 통해
            # imgs와 img_names에 적절한 값을 할당한다.self.prepare
            imgs = samples["image"].type(torch.FloatTensor)
            imgs = imgs.to(torch.device("cuda"))
            img_names = samples["name"]

            # latents = samples["latent"].to(torch.device("cuda"))

            # Gaze Estimator의 옵티마이저를 초기화 한다.
            self.optimizer.zero_grad() 
            
            # 배치 단위의 학습 소요 시간을 파악하기 위해 사용한다.
            timer_data.hold()
            timer_model.tic()
            
            # 배치 단위의 학습 데이터의 파일명을 통해서 GT 시선 라벨을 불러온다.
            # 또한 불러온 라벨을 GPU 연산에 사용하기 위하여 cuda를 붙인다.
            head_batch_label, gaze_batch_label = loadLabel(labels, img_names)
            head_batch_label = head_batch_label.cuda()
            gaze_batch_label = gaze_batch_label.cuda()

            # 순전파 연산을 통해서 모델에 입력을 넣어준다.
            angular_out  = self.model(imgs)
            gaze_loss, angular_error = computeGazeLoss(angular_out, gaze_batch_label)

            total_gaze_loss += gaze_loss
            total_angular_error += angular_error

            # 역전파 연산을 통해 신경망을 업데이트 한다.
            gaze_loss.backward()
            self.optimizer.step()
                
            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t[Average Gaze Loss : {:.4f}]\t{:.1f}+{:.1f}s\t[Average Angular Error:{:.3f}]'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    total_gaze_loss / (self.opt.batch_size * (batch+1)),
                    timer_model.release(),
                    timer_data.release(),total_angular_error / (self.opt.batch_size * (batch+1))))
            timer_data.tic()

        # 1 epoch의 학습 과정을 수행한 후 평균적인 손실 값과 오류 각도를 연산하고 로그를 남길 수 있도록 한다.
        average_gaze_loss =  total_gaze_loss / (self.opt.batch_size * (batch+1))
        average_angular_error = total_angular_error / (self.opt.batch_size * (batch+1))

        print('Train gaze loss : ', float(average_gaze_loss))
        print('Train Angular loss : ', float(average_angular_error))
        
        # 손실 값과 오류 각도에 대한 로그를 남길 수 있도록 한다.
        train_gaze_loss_path = "./experiment/Train_gaze_loss(%s).txt" %self.opt.model
        train_angular_error_path = "./experiment/Train_angular_loss(%s).txt" %self.opt.model
        path_list = [train_gaze_loss_path, train_angular_error_path]
        log_list = [float(average_gaze_loss.item()), float(average_angular_error.item())]

        for i in range(2):
            txt = open(path_list[i], 'a')
            log = str(log_list[i]) + "\n"
            txt.write(log)
            txt.close()

        # 1 epoch의 종료에 따른 클래스 변수의 업데이트
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.step()
      
    def test(self):

        # 1 epoch의 평균 시선 추정 손실값과 오류 각도를 연산하기 위해 정의한다.
        total_gaze_loss = 0
        total_angular_error = 0
        
        # 학습 상황에 따라 현재 epoch와 learning rate를 파악하기 위해 사용한다.
        epoch = self.gaze_model_scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()
        timer_test = utility.timer()

        # 시선 추정 오류를 계산하기 위해 라벨을 불러올 수 있도록 한다.
        label_txt = open("./dataset/MPII_gaze_label.txt" , "r")
        labels = label_txt.readlines()

        with torch.no_grad():

            tqdm_test = tqdm(self.loader_test, ncols=80)
            for _, (sample) in enumerate(tqdm_test):
                
                # Samples라는 배치 단위의 딕셔너리에서 필요한 value 값을 통해
                # imgs와 img_names에 적절한 값을 할당한다.
                img = sample["image"].type(torch.FloatTensor)
                img = img.to(torch.device("cuda"))
                img_name = sample["name"]

                # latent = sample["latent"].to(torch.device("cuda"))

                # 배치 단위의 학습 데이터의 파일명을 통해서 GT 시선 라벨을 불러온다.
                # 또한 불러온 라벨을 GPU 연산에 사용하기 위하여 cuda를 붙인다.
                _, gaze_batch_label = loadLabel(labels, img_name)
                gaze_batch_label = gaze_batch_label.cuda()

                # 순전파 연산을 통해서 모델에 입력을 넣어준다.
                angular_out = self.model(img)
                gaze_loss, angular_error = computeGazeLoss(angular_out, gaze_batch_label)

                total_gaze_loss += gaze_loss
                total_angular_error += angular_error

            self.ckp.log[-1, 0] = total_gaze_loss / len(self.loader_test)
            best = self.ckp.log.min(0)
            self.ckp.write_log(
                '[{}]\tGaze Loss: {:.4f} (Best: {:.4f} @epoch {})'.format(
                    self.opt.data_test,
                    self.ckp.log[-1, 0],
                    best[0][0],
                    best[1][0] + 1
                )
            )

        if not self.opt.test_only:
            Gaze_model_save(self.opt, self.model, epoch, is_best=(best[1][0] + 1 == epoch))

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        # 추론 과정을 수행한 후 평균적인 손실 값과 오류 각도를 연산하고 로그를 남길 수 있도록 한다.
        average_gaze_loss = total_gaze_loss / len(self.loader_test)  
        average_angular_error = total_angular_error / len(self.loader_test)  

        print('Validation gaze loss : ', float(average_gaze_loss.item()))
        print('Validation Angular loss : ', float(average_angular_error.item()))
        
        # 손실 값과 오류 각도에 대한 로그를 남길 수 있도록 한다.
        validation_gaze_loss_path = "./experiment/Validation_gaze_loss(%s).txt" %self.opt.model
        validation_angular_error_path = "./experiment/Validation_angular_loss(%s).txt" %self.opt.model
        path_list = [validation_gaze_loss_path, validation_angular_error_path]
        log_list = [float(average_gaze_loss.item()), float(average_angular_error.item())]

        for i in range(2):
            txt = open(path_list[i], 'a')
            log = str(log_list[i]) + "\n"
            txt.write(log)
            txt.close()



    def step(self):
        self.gaze_model_scheduler.step()

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')

        if len(args) > 1:
            return [a.to(device) for a in args[0]], args[-1].to(device)
        return [a.to(device) for a in args[0]],

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.gaze_model_scheduler.last_epoch
            return epoch >= self.opt.epochs

def Gaze_model_save(opt, gaze_model, epoch, is_best=False):
        path = opt.save
        name = 'gaze_model_latest_' + str(epoch) + ".pt"
        torch.save(
            gaze_model.state_dict(), 
            os.path.join(path, name)
        )
        if is_best:
                torch.save(
                    gaze_model.state_dict(),
                    os.path.join(path, 'gaze_model_best.pt')
                )