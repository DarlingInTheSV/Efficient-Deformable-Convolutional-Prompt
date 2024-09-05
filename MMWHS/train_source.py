import os
import torch
import numpy as np
import argparse, sys, datetime
from config import *
from torch.utils.data import DataLoader
from dataloaders.mmwhs_dataloader import MMWHS_MRDataset
import torch.nn.functional as F
from networks.unet_3d import UNet3D
from torch.nn import CrossEntropyLoss
from MedicalZooPytorch.lib.losses3D import BCEDiceLoss




class TrainSource:
    def __init__(self, config):
        # Save Log and Model
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        self.model_path = os.path.join(config.path_save_model, config.Source_Dataset)  # Save Model
        self.log_path = os.path.join(config.path_save_log, 'train_Source')  # Save Log
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.log_path = os.path.join(self.log_path, time_now + '.log')
        sys.stdout = Logger(self.log_path, sys.stdout)

        # Data Loading
        self.train_data_path = config.train_dataset_root
        self.val_data_path = config.val_dataset_root
        train_dataset = MMWHS_MRDataset(self.train_data_path)
        val_dataset = MMWHS_MRDataset(self.val_data_path)

        self.source_train_loader = DataLoader(dataset=train_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=config.num_workers)
        self.source_val_loader = DataLoader(dataset=val_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=config.num_workers)
        self.image_size = config.image_size

        # Model
        # self.backbone =
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch


        # Optimizer
        self.optim = config.optimizer
        self.lr_scheduler = config.lr_scheduler
        self.lr = config.lr
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.betas = (config.beta1, config.beta2)

        # Training
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        # GPU
        self.device = config.device

        # Initialize the pre-trained model and optimizer
        self.build_model()

        # Print Information
        for arg, value in vars(config).items():
            print(f"{arg}: {value}")
        self.print_network()
        print('***' * 20)

    def build_model(self):
        self.model = UNet3D(1,num_classes=self.out_ch).to(self.device)
        # self.model = ResUnet(resnet=self.backbone, num_classes=self.out_ch, pretrained=True).to(self.device)

        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=True
            )
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                betas=self.betas
            )
        elif self.optim == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                betas=self.betas
            )

        if self.lr_scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-7)
        elif self.lr_scheduler == 'Step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        elif self.lr_scheduler == 'Epoch':
            self.scheduler = EpochLR(self.optimizer, epochs=self.num_epochs, gamma=0.9)
        else:
            self.scheduler = None

    def print_network(self):
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        print("The number of total parameters: {}".format(num_params))

    def run(self):

        best_loss = float('inf')
        criterion = BCEDiceLoss(classes=self.out_ch)
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            total_dice = 0.0
            print("Epoch:{}/{}".format(epoch + 1, self.num_epochs))
            print("Source Pretraining...")
            print("Learning rate: " + str(self.optimizer.param_groups[0]["lr"]))
            num_batches = len(self.source_train_loader)
            for batch, data in enumerate(self.source_train_loader):
                x = data[0].to(self.device)
                y = data[1].to(self.device)
                pred = self.model(x)
                loss, dice_ch = criterion(pred, y)

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # dice = metric_dice_coefficient_3d(pred.detach(), y)
                total_loss += loss.item()
                dice_ch = dice_ch * 100.
                total_dice += dice_ch

            if self.scheduler is not None:
                self.scheduler.step()
            avg_loss = total_loss / num_batches
            class_dice = total_dice / num_batches
            avg_dice = class_dice.mean()
            print("Train ———— Total Loss:{:.8f}".format(avg_loss))
            print(f"Dice : {class_dice}")
            print(f"Avg Dice: {avg_dice}")
            # metrics_y = np.mean(metrics_test, axis=1)
            # print_test_metric = {}
            # for i in range(len(metrics_y)):
            #     print_test_metric[metric_dict[i]] = metrics_y[i]
            # print("Train Metrics Mean: ", print_test_metric)
            # print('*****'*10)
            with torch.no_grad():
                self.model.eval()
                val_batch  = len(self.source_val_loader)
                val_loss = 0.
                val_dice = 0.
                for batch, data in enumerate(self.source_val_loader):
                    x = data[0].to(self.device)
                    y = data[1].to(self.device)
                    pred = self.model(x)
                    loss, dice_ch = criterion(pred, y)
                    val_loss += loss.item()
                    dice_ch = dice_ch * 100.
                    val_dice += dice_ch
                val_avg_loss = val_loss / val_batch
                class_dice = val_dice / val_batch
                avg_dice = class_dice.mean()
                print("Val ———— Total Loss:{:.8f}".format(val_avg_loss))
                print(f"VAL Dice : {class_dice}")
                print(f"VAL Avg Dice: {avg_dice}")


            # Save Model
            if best_loss > val_avg_loss:
                best_loss = val_avg_loss
                best_epoch = (epoch + 1)
                torch.save(self.model.state_dict(), self.model_path + '/' + 'best_pretrain-3d_Unet_CT.pth')

        torch.save(self.model.state_dict(), self.model_path + '/' + 'last_pretrain-3d_Unet.pth')
        print('The best total loss:{} epoch:{}'.format(best_loss, best_epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--Source_Dataset', type=str, default='MMWHS')

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=512)

    # Model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=1)
    parser.add_argument('--out_ch', type=int, default=5) # 0是其他，1-4是要分割的

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='AdamW', help='SGD/Adam/AdamW')
    parser.add_argument('--lr_scheduler', type=str, default='Epoch',
                        help='Cosine/Step/Epoch')   # choose the decrease strategy of lr
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0005)  # weight_decay in SGD
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)  # beta1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)  # beta2 in Adam

    # Training
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)

    # Loss function
    parser.add_argument('--lossmap', type=str, default=['dice', 'bce'])

    # Path
    parser.add_argument('--path_save_log', type=str, default='./logs/')
    parser.add_argument('--path_save_model', type=str, default='./models/')
    parser.add_argument('--train_dataset_root', type=str, default='/home/lsy/PycharmProjects/VPTTA/mmwhs_processed/CT/train')
    parser.add_argument('--val_dataset_root', type=str, default='/home/lsy/PycharmProjects/VPTTA/mmwhs_processed/CT/val')

    # Cuda (default: the first available device)
    parser.add_argument('--device', type=str, default='cuda:0')

    config = parser.parse_args()
    TS = TrainSource(config)
    TS.run()
