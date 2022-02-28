from multiprocessing.spawn import freeze_support
from rt_gene.gaze_estimation_models_pytorch import GazeModel
# from rt_gene.gaze_latent_model import GazeModel
import os
import utility
import data
import loss  
from option import args
from checkpoint import Checkpoint
from trainer import Trainer
import wandb
os.environ['KMP_DUPLICATE_LIB_OK']='True'

utility.set_seed(args.seed)
checkpoint = Checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    gaze_model = GazeModel(args).cuda()
    
    loss = loss.Loss(args, checkpoint)
    t = Trainer(args, loader, gaze_model, loss, checkpoint)

    def main():
        # wandb.init(project='ETH_img_two_infer_latent_eth_with_gd_loss', entity='youngju', config={})
        # wandb.config.update(args) # add all of the arguments as config variables
        
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()

if __name__ == '__main__':  # 중복 방지를 위한 사용
    freeze_support()  # 윈도우에서 파이썬이 자원을 효율적으로 사용하게 만들어준다.
    main()




