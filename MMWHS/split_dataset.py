import os
import shutil
import random


def split_dataset(data_dir, train_dir, val_dir, split_ratio=0.8):
    # Ensure output directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get all image and label files
    image_files = [file for file in os.listdir(data_dir) if file.endswith('_image.nii.gz')]

    # Shuffle files
    random.shuffle(image_files)

    # Split files into train and validation sets
    num_train = int(len(image_files) * split_ratio)
    train_files = image_files[:num_train]
    val_files = image_files[num_train:]

    # Move train files
    for file in train_files:
        image_path = os.path.join(data_dir, file)
        label_file = file.replace('_image.nii.gz', '_label.nii.gz')
        label_path = os.path.join(data_dir, label_file)

        shutil.move(image_path, os.path.join(train_dir, file))
        shutil.move(label_path, os.path.join(train_dir, label_file))

    # Move validation files
    for file in val_files:
        image_path = os.path.join(data_dir, file)
        label_file = file.replace('_image.nii.gz', '_label.nii.gz')
        label_path = os.path.join(data_dir, label_file)

        shutil.move(image_path, os.path.join(val_dir, file))
        shutil.move(label_path, os.path.join(val_dir, label_file))


# 设置路径和比例
data_dir = '/home/lsy/PycharmProjects/VPTTA/mmwhs_processed/MR'
train_dir = '/home/lsy/PycharmProjects/VPTTA/mmwhs_processed/MR/train'
val_dir = '/home/lsy/PycharmProjects/VPTTA/mmwhs_processed/MR/val'
split_ratio = 0.8  # 80% 训练集，20% 验证集

# 执行划分和移动操作
split_dataset(data_dir, train_dir, val_dir, split_ratio)
