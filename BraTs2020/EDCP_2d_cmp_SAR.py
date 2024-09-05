import os
import random

import torch
from tqdm import  tqdm
from MedicalZooPytorch.lib.medzoo.Unet2D import Unet, AdaBN
import numpy as np
import argparse, sys, datetime
from config import Logger, seed_torch
from torch.autograd import Variable

from utils.memory import Memory, Memory2
from utils.prompt3d import Prompt3d
from utils.prompt import Prompt
from utils.metrics import calculate_metrics
from dataloaders.brats2020_dataloader import BratsDataset
from torch.utils.data import DataLoader
# from networks.unet_3d import UNet3D
from MedicalZooPytorch.lib.losses3D.dice_metric import DiceMetric
from MedicalZooPytorch.lib.augment3D import *
from MedicalZooPytorch.lib.losses3D import BCEDiceLoss
from MedicalZooPytorch.lib.losses3D.basic import compute_per_channel_dice
from utils.wrap_meta import set_cal_mseloss, entropy_minmization
import time
from utils.deformable_prompt import DeformablePrompt, normalize_tensor
import torch.nn as nn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from utils.image_op import  save_to_png, save_label_mask, save_prompt_img
from utils.warm_loss import cross_entropy_with_logits, mutual_information_loss, ssim_loss
from torchmetrics.image import StructuralSimilarityIndexMeasure
torch.set_num_threads(1)
class SAM(torch.optim.Optimizer):
    # from https://github.com/davda54/sam
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups






class VPTTA:
    def __init__(self, config):
        # Save Log
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        log_root = os.path.join(config.path_save_log, 'VPTTA')
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        log_path = os.path.join(log_root, time_now + '.log')
        sys.stdout = Logger(log_path, sys.stdout)
        self.writer = SummaryWriter(log_dir="./tensor_log")
        # Data Loading
        self.test_data_path = config.test_dataset_root
        self.Source_Dataset = config.Source_Dataset
        train_dataset = BratsDataset("tta", self.test_data_path, self.Source_Dataset)
        # train_dataset = BratsDataset("train", self.test_data_path, "t1")
        self.target_test_loader = DataLoader(dataset=train_dataset,
                                             batch_size=1,  # batch size = 1 !!
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False,
                                             num_workers=config.num_workers)
        self.image_size = config.image_size
        self.batch_size = config.batch_size
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
        # self.memory_bank = Memory2(size=config.memory_size, dimension=162)
        # Print Information
        # for arg, value in vars(config).items():
        #     print(f"{arg}: {value}")
        # self.print_prompt()
        for param in self.model.parameters():
            param.requires_grad = False
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.requires_grad_(True)
                # force use of batch stats in train and eval modes
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
        print('***' * 20)

    def build_model(self):
        # self.prompt = Prompt(prompt_alpha=self.prompt_alpha, image_size=self.image_size).to(self.device)
        # self.prompt = DeformablePrompt(prompt_alpha=0.001, brats=True)
        # self.prompt = Prompt3d()

        # convert 前后的变化特别大！！！！
        self.model = Unet(1, 3, convert=False).to(self.device)
        # self.model = UNet3D(1, num_classes=self.out_ch, convert=False).to(self.device)
        # self.model = ResUnet(resnet=self.backbone, num_classes=self.out_ch, pretrained=False, newBN=AdaBN, warm_n=self.warm_n).to(self.device)
        checkpoint = torch.load(os.path.join(self.load_model, 'best_pretrain-2d_Unet.pth'))
        self.model.load_state_dict(checkpoint, strict=True)

        if self.optim == 'SGD':
            self.optimizer = SAM(self.model.parameters(), torch.optim.SGD, lr=self.lr, momentum=self.momentum)
        elif self.optim == 'Adam':

            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                # list(self.prompt.parameters()) + list(self.model.parameters()),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay
            )


    def print_prompt(self):
        num_params = 0
        for p in self.prompt.parameters():
            num_params += p.numel()
        print("The number of total parameters: {}".format(num_params))

    def run(self, train_sample):

        # Valid on Target
        metrics_test = [[], [], [], []]
        # metric = BCEDiceLoss(classes=self.out_ch)
        metric = DiceMetric(classes=self.out_ch)
        scores =[]
        stop_grad = False
        skip = 0
        total_loss = 0.0
        total_dice = 0.0
        num_batches = len(self.target_test_loader)
        direct_test = False
        pbar = tqdm(self.target_test_loader)
        total_iter = 0
        total_skip = 0
        for batch, data in enumerate(pbar):

            x = data["image"].to(self.device).permute(1, 0, 2, 3).contiguous()  # 144x1x144x144
            y = data["mask"].to(self.device).float().squeeze().permute(1, 0, 2, 3).contiguous()  # 144x3x144x144
            num_iter = 144 / self.batch_size

            for i in range(int(num_iter)):
                x1 = x[i * self.batch_size: (i + 1) * self.batch_size]
                y1 = y[i * self.batch_size: (i + 1) * self.batch_size]
                if torch.ones(y1.shape).cuda().sum() * 0.01 > y1.sum():  # 去除没有标签的图像，或者标签太少的图像
                    continue
            # name = data[2]
            # max_val, min_val = data["max"], data["min"]
            # 这里交换x x1,x1是未归一化的图像，x是经过归一化的
            # x1 = normalize_tensor(x)
            # x, x1 = x1.clone(), x.clone()

            # if enable_vis:
            #     save_to_png(x[0].detach().cpu(), "/home/lsy/Desktop/111.png")
            #     save_to_png(x2[0].detach().cpu(), "/home/lsy/Desktop/222.png")

                self.model.train()


                for tr_iter in range(self.iters * 1):
                    transform = transforms.Compose([
                        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                        transforms.GaussianBlur(3, 3),

                        # transforms.RandomGrayscale(0.1),
                    ])
                    # x2 = transform(x1)
                    self.model.change_BN_status(new_sample=True)

                    pred_logit1 = self.model(x1)
                    seg_output = torch.sigmoid(pred_logit1) + 1e-6
                    loss = -seg_output * torch.log(seg_output)
                    entro_loss = loss.mean()


                    self.optimizer.zero_grad()  # 这个需要和上面一起吗？

                    entro_loss.backward()

                    self.optimizer.first_step(
                        zero_grad=True)  # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
                    pred_logit = self.model(x1)
                    seg_output = torch.sigmoid(pred_logit) + 1e-6
                    entropys2 = -seg_output * torch.log(seg_output)
                    filter_ids_2 = torch.where(
                        entropys2 < 0.1)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
                    loss_second = entropys2[filter_ids_2].mean(0)
                    # if not np.isnan(loss_second.item()):
                    #     self.ema = update_ema(self.ema,
                    #                           loss_second.item())  # record moving average loss values for model recovery
                    # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
                    loss_second.backward()
                    self.optimizer.second_step(zero_grad=True)

                self.model.change_BN_status(new_sample=False) # 在推理前关掉统计计数

                # Inference
                # self.model.eval()
                # self.prompt.eval()
                with torch.no_grad():

                    pred = self.model(x1)

            # Calculate the metrics

                dice_ch = metric(pred.unsqueeze(2), y1.unsqueeze(2))
                dice_ch = dice_ch * 100.
                total_dice += dice_ch
                total_iter += 1
            if batch % 50 == 0:
                dc = total_dice / total_iter
                print(f"Dice : {dc}")

            # if batch > 380:
            #     direct_test = False
            # if batch % 500 == 0:
            #     plt.plot(scores)
            #     plt.savefig('/home/lsy/Desktop/score.png')


        class_dice = total_dice / total_iter
        avg_dice = class_dice.mean()
        print(f"Dice : {class_dice}")
        print(f"Avg Dice: {avg_dice}")
        print(f"Total train iter : {len(scores)}, percentage : {len(scores) / len(self.target_test_loader)}")
        train_sample.append(len(scores))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--Source_Dataset', type=str, default='t1')
    parser.add_argument('--Target_Dataset', type=list)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=512)

    # Model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=1)
    parser.add_argument('--out_ch', type=int, default=3)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='SGD', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=1e-5)
    # parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)      # beta1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)     # beta2 in Adam.
    parser.add_argument('--weight_decay', type=float, default=0.00)

    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)

    # Hyperparameters in memory bank, prompt, and warm-up statistics
    parser.add_argument('--memory_size', type=int, default=10)
    parser.add_argument('--neighbor', type=int, default=10)
    parser.add_argument('--prompt_alpha', type=float, default=0.01)
    parser.add_argument('--warm_n', type=int, default=5)

    # Path
    parser.add_argument('--path_save_log', type=str, default='./logs')
    parser.add_argument('--model_root', type=str, default='./models')
    parser.add_argument('--test_dataset_root', type=str, default='/home/lsy/Desktop/dataset/BraTS2020')

    # Cuda (default: the first available device)
    parser.add_argument('--device', type=str, default='cuda:0')

    config = parser.parse_args()
    # seed_torch(42)
    config.Target_Dataset = ["t1", "t1ce", "t2", "flair"]
    tt = ["t1", "t1ce", "t2", "flair"]
    # tt = ["flair"]
    target_copy = config.Target_Dataset.copy()
    res = []
    train_sample = []
    start_time = time.time()
    for s in tt:
        config.Source_Dataset = s
        config.Target_Dataset = target_copy.copy()
        config.Target_Dataset.remove(config.Source_Dataset)
        print(f"{config.Source_Dataset} --- >>> {config.Target_Dataset}")
        TTA = VPTTA(config)
        TTA.run(train_sample)
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    print(res)
    print(train_sample)
    print(f"程序运行时间: {elapsed_minutes:.2f} 分钟")