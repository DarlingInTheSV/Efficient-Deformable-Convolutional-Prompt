import os
import torch
import numpy as np
import argparse, sys, datetime
from config import Logger, seed_torch
from torch.autograd import Variable
from networks.convert_3d_BN import AdaBN
from utils.memory import Memory,Memory2
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
        # self.memory_bank = Memory2(size=config.memory_size, dimension=27)
        # Print Information
        # for arg, value in vars(config).items():
        #     print(f"{arg}: {value}")
        # self.print_prompt()
        print('***' * 20)
        for param in self.model.parameters():
            param.requires_grad = False
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm3d):
                module.requires_grad_(True)
                # force use of batch stats in train and eval modes
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

    def build_model(self):
        # self.prompt = Prompt(prompt_alpha=self.prompt_alpha, image_size=self.image_size).to(self.device)
        # self.prompt = DeformablePrompt(prompt_alpha=5, enable_3d=True, ct=True)
        # self.prompt = Prompt3d()

        # convert 前后的变化特别大！！！！
        self.model = UNet3D(1, num_classes=self.out_ch, convert=False).to(self.device)
        # self.model = ResUnet(resnet=self.backbone, num_classes=self.out_ch, pretrained=False, newBN=AdaBN, warm_n=self.warm_n).to(self.device)
        # checkpoint = torch.load(os.path.join(self.load_model, 'best_pretrain-3d_Unet.pth'))
        checkpoint = torch.load(os.path.join(self.load_model, 'best_pretrain-3d_Unet.pth'))
        self.model.load_state_dict(checkpoint, strict=True)
        # self.model.convert()
        if self.optim == 'SGD':
            self.optimizer = SAM(self.model.parameters(), torch.optim.SGD, lr=self.lr, momentum=self.momentum)


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
        for batch, data in enumerate(self.target_test_loader):
            # 对于处理后的标签不是one-hot，白色映射为(0，0)，灰色映射为了(1,0)，黑色映射为了(1,1)
            # 因为OD/OC 分别代表视盘和视杯，盘包括了杯是整体的轮廓，而杯是内部所以采用了特殊的标签，以计算DICE系数，因此损失函数计算时去参考下source train!!!
            x = data[0].to(self.device)
            y = data[1].to(self.device)
            name = data[2]
            # max_val, min_val = data["max"], data["min"]
            # 这里交换x x1,x1是未归一化的图像，x是经过归一化的
            # x1 = normalize_tensor(x)
            # x, x1 = x1.clone(), x.clone()

            # if enable_vis:
            #     save_to_png(x[0].detach().cpu(), "/home/lsy/Desktop/111.png")
            #     save_to_png(x2[0].detach().cpu(), "/home/lsy/Desktop/222.png")
            self.model.train()

            for tr_iter in range(self.iters):

                pred = self.model(x)
                self.optimizer.zero_grad()
                seg_output = torch.sigmoid(pred) + 1e-6
                loss = -seg_output * torch.log(seg_output)
                # if torch.isnan(loss).any():
                #     print("111")
                entro_loss = loss.mean()
                entro_loss.backward()

                self.optimizer.first_step(
                    zero_grad=True)  # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
                pred = self.model(x)
                seg_output = torch.sigmoid(pred) + 1e-6
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

                # self.optimizer.step()
        # self.model.change_BN_status(new_sample=False) # 在推理前关掉统计计数

            # Inference
            self.model.eval()
            with torch.no_grad():
                pred = self.model(x)



            # Calculate the metrics
            dice_ch = metric(pred, y)
            dice_ch = dice_ch * 100.
            total_dice += dice_ch
            # if batch % 500 == 0:
            #     plt.plot(scores)
            #     plt.savefig('/home/lsy/Desktop/score.png')


        class_dice = total_dice / num_batches
        avg_dice = class_dice[1:].mean()
        print(f"Dice : {class_dice}")
        print(f"Avg Dice: {avg_dice}")
        print(f"Total train iter : {len(scores)}, percentage : {len(scores) / len(self.target_test_loader)}")
        train_sample.append(len(scores))

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
    parser.add_argument('--optimizer', type=str, default='SGD', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)      # beta1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)     # beta2 in Adam.
    parser.add_argument('--weight_decay', type=float, default=0.00)

    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)

    # Hyperparameters in memory bank, prompt, and warm-up statistics
    parser.add_argument('--memory_size', type=int, default=5)
    parser.add_argument('--neighbor', type=int, default=5)
    parser.add_argument('--prompt_alpha', type=float, default=0.01)
    parser.add_argument('--warm_n', type=int, default=5)

    # Path
    parser.add_argument('--path_save_log', type=str, default='./logs')
    parser.add_argument('--model_root', type=str, default='./models')
    parser.add_argument('--test_dataset_root', type=str, default='/home/lsy/PycharmProjects/VPTTA/mmwhs_processed/CT/train')

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