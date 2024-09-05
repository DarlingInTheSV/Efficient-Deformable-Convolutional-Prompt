import os
import torch
import numpy as np
import argparse, sys, datetime
from config import Logger, seed_torch
from torch.autograd import Variable
from networks.convert_3d_BN import AdaBN
from utils.memory import Memory
from utils.prompt3d import Prompt3d
from utils.prompt import Prompt
from utils.metrics import calculate_metrics
from dataloaders.mmwhs_dataloader import MMWHS_MRDataset
from torch.utils.data import DataLoader
from networks.unet_3d import UNet3D
from MedicalZooPytorch.lib.losses3D.dice_metric import DiceMetric
from MedicalZooPytorch.lib.losses3D import BCEDiceLoss
from MedicalZooPytorch.lib.losses3D.basic import compute_per_channel_dice
from utils.wrap_meta import set_cal_mseloss, entropy_minmization
import time
from utils.deformable_prompt import DeformablePrompt, normalize_tensor
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils.image_op import  save_to_png, save_label_mask, save_prompt_img
from utils.warm_loss import cross_entropy_with_logits, mutual_information_loss, ssim_loss
torch.set_num_threads(1)





class VPTTA:
    def __init__(self, config):
        # Save Log
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        log_root = os.path.join(config.path_save_log, 'VPTTA')
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        log_path = os.path.join(log_root, time_now + '.log')
        sys.stdout = Logger(log_path, sys.stdout)

        # Data Loading
        self.test_data_path = config.test_dataset_root
        test_dataset = MMWHS_MRDataset(self.test_data_path)
        self.target_test_loader = DataLoader(dataset=test_dataset,
                                             batch_size=config.batch_size,  # batch size = 1 !!
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False,
                                             num_workers=config.num_workers)
        self.image_size = config.image_size

        # Model
        self.load_model = os.path.join(config.model_root, str(config.Source_Dataset))  # Pre-trained Source Model
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch

        # Optimizer
        self.optim = config.optimizer
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.momentum = config.momentum
        self.betas = (config.beta1, config.beta2)

        # GPU
        self.device = config.device

        # Warm-up
        self.warm_n = config.warm_n

        # Prompt
        self.prompt_alpha = config.prompt_alpha
        self.iters = config.iters  # 1 also

        # Initialize the pre-trained model and optimizer
        self.build_model()

        # Memory Bank
        self.neighbor = config.neighbor
        # self.memory_bank = Memory(size=config.memory_size, dimension=self.prompt.data_prompt.weight.numel())
        self.memory_bank = Memory(size=config.memory_size, dimension=27)
        # Print Information
        # for arg, value in vars(config).items():
        #     print(f"{arg}: {value}")

    def build_model(self):
        self.model2 = UNet3D(1, num_classes=self.out_ch, convert=True).to(self.device)
        checkpoint = torch.load(os.path.join(self.load_model, 'best_pretrain-3d_Unet.pth'))
        self.model2.load_state_dict(checkpoint, strict=True)
        self.model = UNet3D(1, num_classes=self.out_ch, convert=False).to(self.device)
        self.model.load_state_dict(checkpoint, strict=True)

        state_dict1 = self.model.state_dict()
        state_dict2 = self.model2.state_dict()
        params_match = True
        for key1, key2 in zip(state_dict1.keys(), state_dict2.keys()):
            if torch.equal(state_dict1[key1], state_dict2[key2]):
                print(f"Parameter '{key1}' matches.")
            else:
                print(f"Parameter '{key1}' does not match.")
                params_match = False

        if params_match:
            print("Both models have identical parameters.")
        else:
            print("Models have different parameters.")

    def run(self, train_sample):

        total_dice = 0.0
        num_batches = len(self.target_test_loader)
        for batch, data in enumerate(self.target_test_loader):
            # 对于处理后的标签不是one-hot，白色映射为(0，0)，灰色映射为了(1,0)，黑色映射为了(1,1)
            # 因为OD/OC 分别代表视盘和视杯，盘包括了杯是整体的轮廓，而杯是内部所以采用了特殊的标签，以计算DICE系数，因此损失函数计算时去参考下source train!!!
            x = data[0].to(self.device)
            y = data[1].to(self.device)
            name = data[2]
            metric = DiceMetric(classes=self.out_ch)
            self.model.eval()
            # self.model2.eval()
            with torch.no_grad():
                pred = self.model(x)
                # pred2 = self.model2(x)

            dice_ch = metric(pred, y)
            # dice_ch2 = metric(pred2, y)
            dice_ch = dice_ch * 100.
            total_dice += dice_ch


        class_dice = total_dice / num_batches
        avg_dice = class_dice.mean()
        print(f"Dice : {class_dice}")
        print(f"Avg Dice: {avg_dice}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--Source_Dataset', type=str, default='MMWHS')
    parser.add_argument('--Target_Dataset', type=list)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=512)

    # Model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=1)
    parser.add_argument('--out_ch', type=int, default=5)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)      # beta1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)     # beta2 in Adam.
    parser.add_argument('--weight_decay', type=float, default=0.00)

    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)

    # Hyperparameters in memory bank, prompt, and warm-up statistics
    parser.add_argument('--memory_size', type=int, default=40)
    parser.add_argument('--neighbor', type=int, default=16)
    parser.add_argument('--prompt_alpha', type=float, default=0.01)
    parser.add_argument('--warm_n', type=int, default=5)

    # Path
    parser.add_argument('--path_save_log', type=str, default='./logs')
    parser.add_argument('--model_root', type=str, default='./models')
    parser.add_argument('--test_dataset_root', type=str, default='/home/lsy/PycharmProjects/VPTTA/mmwhs_processed/MR/train')

    # Cuda (default: the first available device)
    parser.add_argument('--device', type=str, default='cuda:0')

    config = parser.parse_args()
    # seed_torch(42)
    res = []
    train_sample = []
    start_time = time.time()
    TTA = VPTTA(config)
    TTA.run(train_sample)
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    print(res)
    print(train_sample)
    print(f"程序运行时间: {elapsed_minutes:.2f} 分钟")