import os
import random
import shutil

# 定义目标文件夹的根路径
root_dir = '/home/lsy/Desktop/dataset/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

# 找出所有以 BraTS20_Training_开头的文件夹
folders = [name for name in os.listdir(root_dir) if name.startswith('BraTS20_Training_')]

# 随机打乱顺序
random.shuffle(folders)

# 计算训练集和验证集的切分点
split_index = int(0.8 * len(folders))
train_folders = folders[:split_index]
val_folders = folders[split_index:]

# 定义目标文件夹 train 和 val 的路径
train_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'val')

# 创建 train 和 val 文件夹（如果不存在）
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 移动训练集文件夹
for folder in train_folders:
    src = os.path.join(root_dir, folder)
    dst = os.path.join(train_dir, folder)
    shutil.move(src, dst)
    print(f'Moved {folder} to {train_dir}')

# 移动验证集文件夹
for folder in val_folders:
    src = os.path.join(root_dir, folder)
    dst = os.path.join(val_dir, folder)
    shutil.move(src, dst)
    print(f'Moved {folder} to {val_dir}')

print("划分和移动完成。")
