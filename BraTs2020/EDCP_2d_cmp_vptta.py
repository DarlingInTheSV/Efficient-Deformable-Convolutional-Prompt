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
from utils.prompt import Prompt
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from utils.image_op import  save_to_png, save_label_mask, save_prompt_img
from utils.warm_loss import cross_entropy_with_logits, mutual_information_loss, ssim_loss
from torchmetrics.image import StructuralSimilarityIndexMeasure
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
        self.memory_bank = Memory(size=config.memory_size, dimension=196)
        # Print Information
        # for arg, value in vars(config).items():
        #     print(f"{arg}: {value}")
        self.print_prompt()
        print('***' * 20)

    def build_model(self):
        self.prompt = Prompt(prompt_alpha=self.prompt_alpha, image_size=self.image_size, out_channel=1).to(self.device)
        # self.prompt = DeformablePrompt(prompt_alpha=0.001, brats=True)
        # self.prompt = Prompt3d()

        # convert 前后的变化特别大！！！！
        self.model = Unet(1, 3).to(self.device)
        # self.model = UNet3D(1, num_classes=self.out_ch, convert=False).to(self.device)
        # self.model = ResUnet(resnet=self.backbone, num_classes=self.out_ch, pretrained=False, newBN=AdaBN, warm_n=self.warm_n).to(self.device)
        checkpoint = torch.load(os.path.join(self.load_model, 'best_pretrain-2d_Unet.pth'))
        self.model.load_state_dict(checkpoint, strict=True)

        # for param in self.model.parameters():
        #     param.requires_grad = False
        #
        # # 解冻批量归一化层的参数
        # for module in self.model.modules():
        #     if isinstance(module, AdaBN):
        #         for param in module.parameters():
        #             param.requires_grad = True
        # self.model.convert()
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(
                # list(self.prompt.parameters()) + list(self.model.parameters()),
                self.prompt.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.weight_decay
            )
        elif self.optim == 'Adam':

            self.optimizer = torch.optim.Adam(
                self.prompt.parameters(),
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

            # 对于处理后的标签不是one-hot，白色映射为(0，0)，灰色映射为了(1,0)，黑色映射为了(1,1)
            # 因为OD/OC 分别代表视盘和视杯，盘包括了杯是整体的轮廓，而杯是内部所以采用了特殊的标签，以计算DICE系数，因此损失函数计算时去参考下source train!!!
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
                if not direct_test:
                    self.model.train()
                    for param in self.model.parameters():
                        param.requires_grad = False
                    self.prompt.train()



                    # for name, param in self.model.named_parameters():
                    #     if 'bn' in name:  # Check if the parameter belongs to a BN layer
                    #         param.requires_grad = True
                    for module in self.model.modules():
                        if isinstance(module, AdaBN):
                            module.track_running_stats = False
                            # for param in module.parameters():
                            #     param.requires_grad = True

                   # Initialize Prompt
                    if len(self.memory_bank.memory.keys()) >= self.neighbor:
                        _, low_freq = self.prompt(x)
                        init_data, score = self.memory_bank.get_neighbours(keys=low_freq.cpu().numpy(), k=self.neighbor)
                        init_data = init_data[0]
                    else:
                        init_data = torch.ones((1, 1, self.prompt.prompt_size, self.prompt.prompt_size)).data
                    self.prompt.update(init_data)


                    if not stop_grad:
                        for tr_iter in range(self.iters * 1):

                            self.model.change_BN_status(new_sample=True)


                            # with torch.no_grad():
                            prompt_x, _ = self.prompt(x1)
                            pred_logit1 = self.model(prompt_x)
                            # ssim = StructuralSimilarityIndexMeasure().cuda()
                            # aug_prompt_loss2 = 1 - ssim(prompt_x2, prompt_x)
                            # aug_prompt_loss1 = torch.nn.functional.mse_loss(prompt_x, prompt_x2)
                            # aug_prompt_loss = ssim_loss(prompt_x, prompt_x2)

                            times, bn_loss = 0, 0

                            #
                            self.optimizer.zero_grad()  # 这个需要和上面一起吗？

                            for nm, m in self.model.named_modules():
                                if isinstance(m, AdaBN):
                                    bn_loss += m.bn_loss
                                    times += 1
                            loss = bn_loss / times
                            loss_tt = loss# reg 0.8    loss: 0.02
                            # loss_tt = aug_prompt_loss + consistency_loss + loss
                            # self.writer.add_scalar("aug_prompt_loss1", aug_prompt_loss1.item(),total_iter)
                            # self.writer.add_scalar("aug_prompt_loss2", aug_prompt_loss2.item(), total_iter)
                            # self.writer.add_scalar("consistency_loss", consistency_loss.item(), total_iter)
                            # self.writer.add_scalar("BN loss", loss.item(), total_iter)
                            loss_tt.backward()
                            self.optimizer.step()
                    self.model.change_BN_status(new_sample=False) # 在推理前关掉统计计数

                    # Inference
                    # self.model.eval()
                    # self.prompt.eval()
                    with torch.no_grad():
                        prompt_x, low_freq = self.prompt(x1)
                        pred = self.model(prompt_x)

                        # # 可视化标签
                        # predicted_labels = torch.argmax(pred, dim=1)  # 形状为 1x144x144x144
                        #
                        # # 将 PyTorch 张量转换为 Numpy 数组
                        # predicted_labels_np = predicted_labels.squeeze().cpu().numpy().astype(np.int16)  # 去除批量维度，转换为 Numpy 数组
                        #
                        # # 将 Numpy 数组保存为 Nifti 格式的文件
                        # output_filename = '/home/lsy/Desktop/inference/only/' + name[0].replace('_image.nii.gz', '_label.nii.gz')
                        # import nibabel as nib
                        # nifti_img = nib.Nifti1Image(predicted_labels_np, affine=np.eye(4))  # 创建 Nifti 图像对象
                        # nib.save(nifti_img, output_filename)  # 保存为 nii.gz 文件

                        # if batch > 0.2 * len(self.target_test_loader):
                        #     save_prompt_img(x[0], "/home/lsy/Desktop/ori.png", True)
                        #     save_prompt_img(prompt[0], "/home/lsy/Desktop/pp.png", True)
                        #     save_prompt_img(prompt_x[0], "/home/lsy/Desktop/prom.png",False, max_val, min_val)

                    # Update the Memory Bank
                    if not stop_grad:
                        self.memory_bank.push(keys=low_freq.cpu().numpy(), logits=self.prompt.data_prompt.detach().cpu().numpy())

                    else:
                        skip -= 1

                else:
                    # self.model.eval()
                    self.prompt.eval()
                    with torch.no_grad():
                        pred = self.model(x1)

                    # --------- vis ----------------
                    # import nibabel as nib
                    # x = x.squeeze(0).squeeze(0).cpu().numpy()
                    # output_filename = '/home/lsy/Desktop/1233_image.nii.gz'
                    # nifti_img = nib.Nifti1Image(x, affine=np.eye(4))  # 创建 Nifti 图像对象
                    # nib.save(nifti_img, output_filename)  # 保存为 nii.gz 文件
                    # # 可视化标签
                    # mask = np.zeros((144, 144, 144))
                    # ly = y.cpu().numpy()
                    # mask_WT = ly[0,0]
                    # mask_TC = ly[0,1]
                    # mask_ET = ly[0,2]
                    # # 还原 mask_WT
                    # mask[mask_WT == 1] = 2
                    #
                    # # 还原 mask_TC
                    # mask[mask_TC == 1] = 4
                    # # mask[mask_TC == 0] = 1  # 需要处理 mask_TC == 0 的情况，因为原来的代码中 mask_TC == 2 时被置为 0
                    #
                    # # 还原 mask_ET
                    # mask[mask_ET == 1] = 1
                    # # mask[mask_ET == 0] = 2  # 需要处理 mask_ET == 0 的情况，因为原来的代码中 mask_ET == 2 时被置为 0
                    # # predicted_labels = torch.argmax(pred, dim=1)  # 形状为 1x144x144x144
                    # # y = torch.argmax(y, dim=1)
                    # # # 将 PyTorch 张量转换为 Numpy 数组
                    # # predicted_labels_np = predicted_labels.squeeze().cpu().numpy().astype(
                    # #     np.int16)  # 去除批量维度，转换为 Numpy 数组
                    # # yl = y.squeeze().cpu().numpy().astype(
                    # #     np.int16)
                    # # 将 Numpy 数组保存为 Nifti 格式的文件
                    # # output_filename = '/home/lsy/Desktop/inference/only/' + name[0].replace('_image.nii.gz',
                    # #                                                                         '_label.nii.gz')
                    #
                    # pred_mask = np.zeros((144, 144, 144))
                    # m = torch.nn.Sigmoid()
                    # pred = m(pred)
                    # pred = (pred >= 0.5)
                    # pred = pred.cpu().numpy()
                    # mask_WT = pred[0, 0]
                    # mask_TC = pred[0, 1]
                    # mask_ET = pred[0, 2]
                    # # 还原 mask_WT
                    # pred_mask[mask_WT == 1] = 2
                    #
                    # # 还原 mask_TC
                    # pred_mask[mask_TC == 1] = 4
                    # # pred_mask[mask_TC == 0] = 1  # 需要处理 mask_TC == 0 的情况，因为原来的代码中 mask_TC == 2 时被置为 0
                    #
                    # # 还原 mask_ET
                    # pred_mask[mask_ET == 1] = 1
                    # # pred_mask[mask_ET == 0] = 2
                    #
                    #
                    # output_filename = '/home/lsy/Desktop/1233_label.nii.gz'
                    # lb_output_filename = '/home/lsy/Desktop/1233_label_gt.nii.gz'
                    # nifti_img = nib.Nifti1Image(pred_mask, affine=np.eye(4))  # 创建 Nifti 图像对象
                    # nib.save(nifti_img, output_filename)  # 保存为 nii.gz 文件
                    # nifti_img = nib.Nifti1Image(mask, affine=np.eye(4))  # 创建 Nifti 图像对象
                    # nib.save(nifti_img, lb_output_filename)  # 保存为 nii.gz 文件

                    # --------- vis ----------------

            # Calculate the metrics

                dice_ch = metric(pred.unsqueeze(2), y1.unsqueeze(2))
                dice_ch = dice_ch * 100.
                total_dice += dice_ch
                total_iter += 1
            if batch % 50 == 0:
                dc = total_dice / total_iter
                print(f"Dice : {dc}")
                self.writer.add_scalar("Dice1", dc[0], total_iter)
                self.writer.add_scalar("Dice2", dc[1], total_iter)
                self.writer.add_scalar("Dice3", dc[2], total_iter)
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
    parser.add_argument('--image_size', type=int, default=144)

    # Model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=1)
    parser.add_argument('--out_ch', type=int, default=3)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=0.05)
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
    parser.add_argument('--prompt_alpha', type=float, default=0.1)
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