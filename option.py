import argparse
import utility
import numpy as np
import wandb

wandb.login()

parser = argparse.ArgumentParser(description='Gaze')

parser.add_argument('--n_threads', type=int, default=0,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--data_train', type=str, default='img/inferenced_eth_with_ffhq',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='img/inferenced_eth_with_ffhq_validation',
                    help='test dataset name')
# 로그파일을 구분하기 위해 모델명을 입력하세요.
parser.add_argument('--model', type=str, default='resnet18',
                    help='model name')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trainedf model directory')
parser.add_argument('--test_every', type=int, default = 10196, # 10196 epoch
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,        
                    help='input batch size for training')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--loss', type=str, default='1*MSE',
                    help='loss function configuration, L1|MSE')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--eta_min', type=float, default=1e-7,
                    help='eta_min lr')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')
parser.add_argument('--save', type=str, default='./experiment/mpii_gd_loss_256/',
                    help='file name to save')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--label', type=str, default='./dataset/eth_label_dict.pickle',
                    help='load the label file as a pickle file')


args = parser.parse_args()

