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

        # Initialize the pre-trained model and optimizer
        self.build_model()

        # Memory Bank
        self.neighbor = config.neighbor
        # self.memory_bank = Memory(size=config.memory_size, dimension=self.prompt.data_prompt.weight.numel())
        self.memory_bank = Memory(size=config.memory_size)
        # Print Information
        # for arg, value in vars(config).items():
        #     print(f"{arg}: {value}")
        self.print_prompt()
        print('***' * 20)

    def build_model(self):
        # self.prompt = Prompt(prompt_alpha=self.prompt_alpha, image_size=self.image_size).to(self.device)
        self.prompt = DeformablePrompt(prompt_alpha=0.1)
        self.model = ResUnet(resnet=self.backbone, num_classes=self.out_ch, pretrained=False, newBN=AdaBN, warm_n=self.warm_n).to(self.device)
        checkpoint = torch.load(os.path.join(self.load_model, 'last-Res_Unet.pth'))
        self.model.load_state_dict(checkpoint, strict=False)
        # self.model.convert()
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(
                list(self.prompt.parameters()) + list(self.model.parameters()),
                lr=self.lr,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.weight_decay
            )
            self.optimizer2 = torch.optim.SGD(
                [param for meta_part in self.model.res.meta_parts for param in meta_part.parameters()],
                lr=self.lr*0.1,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.weight_decay
            )
        elif self.optim == 'Adam':
            # self.optimizer = torch.optim.Adam(
            #     list(self.prompt.parameters()) + list(self.model.parameters()),
            #     lr=self.lr,
            #     betas=self.betas,
            #     weight_decay=self.weight_decay
            # )

            self.optimizer = torch.optim.Adam(
                self.prompt.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay
            )

            # self.optimizer = torch.optim.Adam(
            #     list(self.prompt.parameters()),
            #     lr=self.lr * 1,
            #     betas=self.betas,
            #     weight_decay=self.weight_decay
            # )
            # self.optimizer2 = torch.optim.Adam(
            #     [param for meta_part in self.model.res.meta_parts for param in meta_part.parameters()] + \
            #     [param for meta_part in [self.model.up_meta1, self.model.up_meta2] for param in meta_part.parameters()],
            #     lr=self.lr * 0.1 ,
            #     betas=self.betas,
            #     weight_decay=self.weight_decay
            # )

    def print_prompt(self):
        num_params = 0
        for p in self.prompt.parameters():
            num_params += p.numel()
        print("The number of total parameters: {}".format(num_params))

    def run(self, res, train_sample):
        metric_dict = ['Disc_Dice', 'Disc_ASD', 'Cup_Dice', 'Cup_ASD']

        # Valid on Target
        metrics_test = [[], [], [], []]
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.GaussianBlur(3, 3),
            # transforms.RandomGrayscale(0.1),
        ])
        warm_meta = 20
        warm_iter = 0
        enable_vis = False # 输出图像及预测mask
        scores =[]
        stop_grad = False
        skip = 0

        for param in self.model.parameters():
            param.requires_grad = False
        # for module in self.model.modules():
        #     if isinstance(module, AdaBN):
        #         module.requires_grad_(True)
        #         # force use of batch stats in train and eval modes
        #         module.track_running_stats = False
        #         module.running_mean = None
        #         module.running_var = None

        for batch, data in enumerate(self.target_test_loader):
            # 对于处理后的标签不是one-hot，白色映射为(0，0)，灰色映射为了(1,0)，黑色映射为了(1,1)
            # 因为OD/OC 分别代表视盘和视杯，盘包括了杯是整体的轮廓，而杯是内部所以采用了特殊的标签，以计算DICE系数，因此损失函数计算时去参考下source train!!!
            x, y = data['data'], data['mask']
            x = torch.from_numpy(x).to(dtype=torch.float32)
            y = torch.from_numpy(y).to(dtype=torch.float32)
            max_val = data["max_val"][0]
            min_val = data["min_val"][0]
            x, y = Variable(x).to(self.device), Variable(y).to(self.device)

            # max_val, min_val = data["max"], data["min"]
            # 这里交换x x1,x1是未归一化的图像，x是经过归一化的
            # x1 = normalize_tensor(x)
            # x, x1 = x1.clone(), x.clone()

            # if enable_vis:
            #     save_to_png(x[0].detach().cpu(), "/home/lsy/Desktop/111.png")
            #     save_to_png(x2[0].detach().cpu(), "/home/lsy/Desktop/222.png")

            self.model.eval()
            self.prompt.train()
            self.model.change_BN_status(new_sample=True)

           # Initialize Prompt
            if len(self.memory_bank.memory.keys()) >= self.neighbor:
                # self.prompt.setAlpha(0.001)  # 这个就是 select update，因此使用memory bank
                _, offset = self.prompt(x)   #
                # 这个加权的memory bank是一个重点！！！不加他会掉点非常多
                if not stop_grad or skip == 0:
                    init_data, score = self.memory_bank.get_neighbours(keys=offset.cpu().detach().numpy(), k=self.neighbor)
                    self.prompt.update(init_data)
                    if score >= 0.96:
                        # scores.append(score)
                        # stop_grad = False
                        # for param in self.prompt.parameters():
                        #     param.requires_grad = True
                        stop_grad = True
                        skip = 1
                        for param in self.prompt.parameters():
                            param.requires_grad = False
                    else:
                        scores.append(score)
                        stop_grad = False
                        for param in self.prompt.parameters():
                            param.requires_grad = True
                #
            else:
                # pass
                self.prompt.setAlpha(0.001)
                # 讲故事，这个是prompt continual warm-up，不用bank，先让prompt从随机值变得有意义
            # self.prompt.setAlpha(0.001)
            # warm-up meta layer
            # for i in range(1):
                # for meta_part in self.model.res.meta_parts:
                #     for param in meta_part.parameters():
                #         param.requires_grad = True

            # if warm_iter < warm_meta:
            #     x2 = transform(x) # 这里要求必须是归一化后的吗？试试未归一化的x1?
            #     # self.optimizer2.zero_grad()
            #     pred_logit2, _, __ = self.model(x2)
            #     # with torch.no_grad():
            #     #     pred_logit1, _, __ = self.model(x)
            #     pred_logit1, _, __ = self.model(x)
                # if enable_vis:
                #     save_label_mask(pred_logit1, "/home/lsy/Desktop/111_label.png")
                #     save_label_mask(pred_logit2, "/home/lsy/Desktop/222_label.png")
                #     save_label_mask(y, "/home/lsy/Desktop/label.png", True)
            #
            #
                # consistency_loss = cross_entropy_with_logits(pred_logit1, pred_logit2)
                # consistency_loss.backward()
                # # self.optimizer2.step()
                # warm_iter += 1
            # else:
            #     for meta_part in self.model.res.meta_parts:
            #         for param in meta_part.parameters():
            #             param.requires_grad = False

            # for meta_part in self.model.res.meta_parts:
            #     for param in meta_part.parameters():
            #         param.requires_grad = False
            # Train Prompt for n iters (1 iter in our VPTTA)

            if not stop_grad:
                for tr_iter in range(self.iters * 1):
                    prompt_x, _= self.prompt(x)
                    # if warm_iter < warm_meta:
                    #     # 增强前后prompt一致性
                    #     prompt_x2, _ = self.prompt(x2) # 过了warm-up后x2一直固定了！！！！
                    #     aug_prompt_loss = nn.functional.mse_loss(prompt_x, prompt_x2)
                    #     aug_prompt_loss = ssim_loss(prompt_x, prompt_x2)

                    times, bn_loss = 0, 0

                    # set_cal_mseloss(self.model, True)
                    # self.model.cal_mseloss = True
                    pred_logit, fea, head_input = self.model(prompt_x)
                    loss_reg_all = 0.
                    gamma = 1
                    self.optimizer.zero_grad()
                    # self.optimizer2.zero_grad()
                    # for i, encoder in enumerate(self.model.res.encoders):
                    #     reg_loss = encoder.btsloss * gamma
                    #     loss_reg_all += reg_loss
                        # reg_loss.backward()
                    # entro_loss = entropy_minmization(pred_logit, 0.0001)
                    # loss_reg_all += self.model.btsloss

                    for nm, m in self.model.named_modules():
                        if isinstance(m, AdaBN):
                            bn_loss += m.bn_loss
                            times += 1
                    loss = bn_loss / times
                    loss_tt = loss
                    # seg_output = torch.sigmoid(pred_logit)
                    # loss_e = -seg_output * torch.log(seg_output)
                    # entro_loss = loss_e.mean()
                    #
                    # loss_tt = loss + entro_loss  # reg 0.8    loss: 0.02

                    # if warm_iter < warm_meta:
                    #     loss_tt += aug_prompt_loss
                    # 发现，没有reg不行，但是调整权重结果不变？
                    loss_tt.backward()
                    #
                    #
                    # loss.backward()
                    self.optimizer.step()
                    # self.optimizer2.step()
                    # set_cal_mseloss(self.model, False)
                    # self.model.cal_mseloss = False
            # else:
            #     with torch.no_grad():
            #         pred_logit, fea, head_input = self.model(prompt_x)
            self.model.change_BN_status(new_sample=False) # 在推理前关掉统计计数

            # Inference
            self.model.eval()
            self.prompt.eval()
            with torch.no_grad():
                prompt_x, weights, prompt = self.prompt(x, True)
                pred_logit, fea, head_input = self.model(prompt_x)
                # if batch > 0.4 * len(self.target_test_loader):
                #     save_prompt_img(x[0], "/home/lsy/Desktop/ori.png", True)
                #     save_prompt_img(prompt[0], "/home/lsy/Desktop/prompt.png", True)
                #     # save_prompt_img(prompt_x[0], "/home/lsy/Desktop/prompt.png",False, False, max_val, min_val)
                #     save_prompt_img(prompt_x[0], "/home/lsy/Desktop/adapted.png", True, True, max_val, min_val)


            # Update the Memory Bank
            if not stop_grad:
                self.memory_bank.push(keys=weights.detach().cpu().numpy(), logits=self.prompt.data_prompt.weight.detach().cpu().numpy())
            else:
                skip -= 1
            # Calculate the metrics
            seg_output = torch.sigmoid(pred_logit) # 映射到0-1之间，这里可以发现pred_logit数值已经有了-16等比较大的数值
            metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
            for i in range(len(metrics)):
                assert isinstance(metrics[i], list), "The metrics value is not list type."
                metrics_test[i] += metrics[i]
            # if batch % 500 == 0:
            #     plt.plot(scores)
            #     plt.savefig('/home/lsy/Desktop/score.png')


        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
        print("Test Metrics: ", print_test_metric_mean)
        print('Mean Dice:', (print_test_metric_mean['Disc_Dice'] + print_test_metric_mean['Cup_Dice']) / 2)
        res.append((print_test_metric_mean['Disc_Dice'] + print_test_metric_mean['Cup_Dice']) / 2)
        print(f"Total train iter : {len(scores)}, percentage : {len(scores) / len(self.target_test_loader)}")
        train_sample.append(len(scores))

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
    parser.add_argument('--model_root', type=str, default='./model3')
    parser.add_argument('--dataset_root', type=str, default='/home/lsy/PycharmProjects/VPTTA/OPTIC/data')

    # Cuda (default: the first available device)
    parser.add_argument('--device', type=str, default='cuda:0')

    config = parser.parse_args()
    # seed_torch(42)
    # 数据集数量分布： RIM_ONE_r3:159  REFUGE:400 ORIGA: 650 REFUGE_Valid:800 Drishti_GS:101
    config.Target_Dataset = ['RIM_ONE_r3', 'REFUGE', 'ORIGA', 'REFUGE_Valid', 'Drishti_GS']
    tt = ['RIM_ONE_r3', 'REFUGE', 'ORIGA', 'REFUGE_Valid', 'Drishti_GS']
    # tt = ['Drishti_GS']
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
        TTA.run(res, train_sample)
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    print(res)
    print(train_sample)
    print(f"程序运行时间: {elapsed_minutes:.2f} 分钟")