import os
import torch
import numpy as np
import argparse, sys, datetime
from config import Logger, seed_torch
from torch.autograd import Variable
from utils.convert import AdaBN
from utils.memory import Memory
from utils.prompt import Prompt
from utils.metrics import calculate_metrics
from networks.ResUnet_TTA import ResUnet
from torch.utils.data import DataLoader
from dataloaders.OPTIC_dataloader import OPTIC_dataset
from dataloaders.transform import collate_fn_wo_transform
from dataloaders.convert_csv_to_list import convert_labeled_list
from utils.wrap_meta import set_cal_mseloss, entropy_minmization
import time
from utils.deformable_prompt import DeformablePrompt, normalize_tensor
import torch.nn as nn
import torchvision.transforms as transforms
from utils.image_op import  save_to_png, save_label_mask
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
        target_test_csv = []
        for target in config.Target_Dataset:
            if target != 'REFUGE_Valid':
                target_test_csv.append(target + '_train.csv') # we use both tain and test splits as unlabelled samples
                target_test_csv.append(target + '_test.csv')
            else:
                target_test_csv.append(target + '.csv')
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)
        target_test_dataset = OPTIC_dataset(config.dataset_root, ts_img_list, ts_label_list,
                                            config.image_size, img_normalize=True)   # !!! 这里试一下prompt加到原图像上
        self.target_test_loader = DataLoader(dataset=target_test_dataset,
                                             batch_size=config.batch_size,  # batch size = 1 !!
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False,
                                             collate_fn=collate_fn_wo_transform,
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
        # 直接测试
        self.direct_test = True

        # Initialize the pre-trained model and optimizer
        self.build_model()

        # Memory Bank
        self.neighbor = config.neighbor

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
        trainable_paras = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {trainable_paras}")
        print('***' * 20)


    def build_model(self):

        if self.direct_test:
            convert = False
        else:
            convert = True
        self.model = ResUnet(convert=convert, resnet=self.backbone, num_classes=self.out_ch, pretrained=False, newBN=AdaBN, warm_n=self.warm_n).to(self.device)
        checkpoint = torch.load(os.path.join(self.load_model, 'last-Res_Unet.pth'))
        self.model.load_state_dict(checkpoint, strict=False)
        # self.model.convert()
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(
                list(self.model.parameters()),
                lr=self.lr,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.weight_decay
            )
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(
                list(self.model.parameters()),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay
            )



    def run(self, res):
        metric_dict = ['Disc_Dice', 'Disc_ASD', 'Cup_Dice', 'Cup_ASD']

        # Valid on Target
        metrics_test = [[], [], [], []]
        for long_term_test in range(1):
            for batch, data in enumerate(self.target_test_loader):
                x, y = data['data'], data['mask']
                x = torch.from_numpy(x).to(dtype=torch.float32)
                y = torch.from_numpy(y).to(dtype=torch.float32)

                x, y = Variable(x).to(self.device), Variable(y).to(self.device)
                self.model.train()
                self.model.change_BN_status(new_sample=False)


                # set_cal_mseloss(self.model, True)
                pred_logit, fea, head_input = self.model(x)
                # loss_reg_all = 0.
                # gamma = 1
                self.optimizer.zero_grad()
                # # self.optimizer2.zero_grad()
                # for i, encoder in enumerate(self.model.res.encoders):
                #     reg_loss = encoder.btsloss * gamma
                #     loss_reg_all += reg_loss
                    # reg_loss.backward()
                seg_output = torch.sigmoid(pred_logit) + 1e-6
                loss = -seg_output * torch.log(seg_output)
                # if torch.isnan(loss).any():
                #     print("111")
                entro_loss = loss.mean()
                # seg_output[seg_output >= 0.5] = 1
                # seg_output[seg_output < 0.5] = 0

                # torch.nonzero(torch.isnan(loss))
                # entro_loss = entropy_minmization(pred_logit, 0.1)

                # loss_tt = loss_reg_all + loss + aug_prompt_loss   # reg 0.8    loss: 0.02
                # 发现，没有reg不行，但是调整权重结果不变？
                entro_loss.backward()
                #
                #
                # loss.backward()
                self.optimizer.step()
                # self.optimizer2.step()
                # set_cal_mseloss(self.model, False)
                self.model.change_BN_status(new_sample=False) # 在推理前关掉统计计数

                # Inference
                self.model.eval()
                # self.prompt.eval()
                with torch.no_grad():
                    # prompt_x, low_freq = self.prompt(x)
                    pred_logit, fea, head_input = self.model(x)


                # Calculate the metrics
                seg_output = torch.sigmoid(pred_logit) # 映射到0-1之间，这里可以发现pred_logit数值已经有了-16等比较大的数值
                metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
                for i in range(len(metrics)):
                    assert isinstance(metrics[i], list), "The metrics value is not list type."
                    metrics_test[i] += metrics[i]
            print(f"Time {long_term_test + 1}: {np.mean(metrics_test, axis=1)}")

        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
        print("Test Metrics: ", print_test_metric_mean)
        print('Mean Dice:', (print_test_metric_mean['Disc_Dice'] + print_test_metric_mean['Cup_Dice']) / 2)
        res.append((print_test_metric_mean['Disc_Dice'] + print_test_metric_mean['Cup_Dice']) / 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--Source_Dataset', type=str, default='Drishti_GS',
                        help='RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS')
    parser.add_argument('--Target_Dataset', type=list)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=512)

    # Model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=2)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=1e-4)
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
    parser.add_argument('--prompt_alpha', type=float, default=0.008)
    parser.add_argument('--warm_n', type=int, default=5)

    # Path
    parser.add_argument('--path_save_log', type=str, default='./logs')
    parser.add_argument('--model_root', type=str, default='./model3')
    parser.add_argument('--dataset_root', type=str, default='/home/lsy/PycharmProjects/VPTTA/OPTIC/data')

    # Cuda (default: the first available device)
    parser.add_argument('--device', type=str, default='cuda:0')

    config = parser.parse_args()
    seed_torch(43)
    # 数据集数量分布： RIM_ONE_r3:159  REFUGE:400 ORIGA: 650 REFUGE_Valid:800 Drishti_GS:101
    config.Target_Dataset = ['RIM_ONE_r3', 'REFUGE', 'ORIGA', 'REFUGE_Valid', 'Drishti_GS']
    tt = ['RIM_ONE_r3', 'REFUGE', 'ORIGA', 'REFUGE_Valid', 'Drishti_GS']
    # tt = ['Drishti_GS']
    target_copy = config.Target_Dataset.copy()
    res = []

    start_time = time.time()
    for s in tt:
        config.Source_Dataset = s
        config.Target_Dataset = target_copy.copy()
        config.Target_Dataset.remove(config.Source_Dataset)
        print(f"{config.Source_Dataset} --- >>> {config.Target_Dataset}")
        TTA = VPTTA(config)
        TTA.run(res)
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    print(res)
    print(f"程序运行时间: {elapsed_minutes:.2f} 分钟")