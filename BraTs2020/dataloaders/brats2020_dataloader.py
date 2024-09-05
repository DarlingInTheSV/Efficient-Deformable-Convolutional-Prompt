import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import random
import re

def sort_by_number(path):
    # 使用正则表达式提取路径中的数字序号
    match = re.search(r'BraTS20_Training_(\d+)', path)
    if match:
        return int(match.group(1))
    else:
        return 0  # 如果找不到数字序号，默认返回0



class BratsDataset(Dataset):
    def __init__(self, phase, data_root, modality):

        self.phase = phase
        self.root = str(data_root)
        self.training_path = self.root + '/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/train'
        self.testing_path = self.root + '/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/val'
        self.modality = modality
        # self.augmentations = get_augmentations(phase)
        if self.phase == "train":
            self.phase_path = self.training_path
        elif self.phase == "val":
            self.phase_path = self.testing_path
        elif self.phase == "tta":
            self.phase_path = [self.training_path, self.testing_path]
        assert modality in ["t1", "t1ce", "t2", "flair"]
        if self.phase == "tta":
            t1_path = sorted(glob.glob(os.path.join(self.phase_path[0], '*/*t1.nii')) + glob.glob(
                os.path.join(self.phase_path[1], '*/*t1.nii')),  key=sort_by_number)
            t1ce_path = sorted(glob.glob(os.path.join(self.phase_path[0], '*/*t1ce.nii')) + glob.glob(
                os.path.join(self.phase_path[1], '*/*t1ce.nii')), key=sort_by_number)
            t2_path = sorted(glob.glob(os.path.join(self.phase_path[0], '*/*t2.nii')) + glob.glob(
                os.path.join(self.phase_path[1], '*/*t2.nii')), key=sort_by_number)
            flair_path = sorted(glob.glob(os.path.join(self.phase_path[0], '*/*_flair.nii')) + glob.glob(
                os.path.join(self.phase_path[1], '*/*_flair.nii')), key=sort_by_number)
            label_path = sorted(glob.glob(os.path.join(self.phase_path[0], '*/*_seg.nii')) + glob.glob(
                os.path.join(self.phase_path[1], '*/*_seg.nii')), key=sort_by_number)
            dic = {
                "t1": t1_path,
                "t1ce": t1ce_path,
                "t2": t2_path,
                "flair": flair_path
            }
            self.img_path = []
            for m in ["t1", "t1ce", "t2", "flair"]:
                if self.modality == m:
                    continue
                else:
                    self.img_path.append(dic[m])
            self.img_path = [item for sublist in self.img_path for item in sublist]
            self.label_path = label_path * 3
        else:
            if modality == "t1":
                img_path = sorted(glob.glob(os.path.join(self.phase_path, '*/*t1.nii')))
            elif modality == "t1ce":
                img_path = sorted(glob.glob(os.path.join(self.phase_path, '*/*t1ce.nii')))
            elif modality == "t2":
                img_path = sorted(glob.glob(os.path.join(self.phase_path, '*/*t2.nii')))
            elif modality == "flair":
                img_path = sorted(glob.glob(os.path.join(self.phase_path, '*/*_flair.nii')))
            label_path = sorted(glob.glob(os.path.join(self.phase_path, '*/*_seg.nii')))
            l = list(zip(img_path, label_path))
            random.seed(43)
            random.shuffle(l)
            self.img_path, self.label_path = zip(*l)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):

        # load all modalities
        img = self.load_img(self.img_path[idx]).astype(np.float32)  # .transpose(2, 0, 1)

        # if self.is_resize:
        #     img = self.resize(img)

        img = self.normalize(img)
        img = np.moveaxis(img, (0, 1, 2), (2, 1, 0))

        mask = self.load_img(self.label_path[idx]).astype(np.int32)

        # if self.is_resize:
        #     mask = self.resize(mask)
        #     mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
        #     mask = np.clip(mask, 0, 1)
        mask = self.preprocess_mask_labels(mask)

        # 切片，适应网络尺寸 144,144,144
        img = img[4:148, 50:194, 50:194]
        mask = mask[:, 4:148, 50:194, 50:194]
        # img = img[50:130]
        # mask = mask[:, 50:130]


        # mask = np.moveaxis(mask, (0, 1, 2), (2, 1, 0))
        # mask = mask[4:148, 50:194, 50:194]

        # img = img[4:168, 50:210, 50:210]
        # mask = mask[:, 4:168, 50:210, 50:210]
        # augmented = self.augmentations(image=img.astype(np.float32),
        #                                mask=mask.astype(np.float32))

        # output_filename = '/home/lsy/Desktop/1233_image.nii.gz'
        # nifti_img = nib.Nifti1Image(img, affine=np.eye(4))  # 创建 Nifti 图像对象
        # nib.save(nifti_img, output_filename)  # 保存为 nii.gz 文件
        #
        # output_filename = '/home/lsy/Desktop/1233_label_gt.nii.gz'
        # nifti_img = nib.Nifti1Image(mask, affine=np.eye(4))  # 创建 Nifti 图像对象
        # nib.save(nifti_img, output_filename)  # 保存为 nii.gz 文件
        # img = augmented['image']
        # mask = augmented['mask']
        # import matplotlib.pyplot as plt
        # plt.imshow(img[:, 70, :] * 255)
        # plt.savefig("/home/lsy/Desktop/yy.png")
        # plt.imshow(mask[0, :, 70, :] * 255)
        # plt.savefig("/home/lsy/Desktop/yy2.png")

        return {
            "image": img,  # 第一个维度是横向切片，第二个是从前向后竖着切，第三个维度是从左向右竖着切
            "mask": mask,
        }

    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        data = np.rot90(data)
        return data

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    # def resize(self, data: np.ndarray):
    #     data = resize(data, (78, 120, 120), preserve_range=True)
    #     return data

    def preprocess_mask_labels(self, mask: np.ndarray):

        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        return mask


if __name__ == "__main__":
    dataset = BratsDataset('tta', "/home/lsy/Desktop/dataset/BraTS2020", "t1")
    dataloader = DataLoader(dataset,
                            batch_size=10,  # batch size = 1 !!
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=10)
    for batch, data in enumerate(dataloader):
        img = data["image"]
        label = data["mask"]
        print(img.shape)
        print(label.shape)
