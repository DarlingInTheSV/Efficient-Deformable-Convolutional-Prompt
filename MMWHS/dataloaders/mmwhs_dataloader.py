import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import re


class MMWHS_MRDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = [file for file in os.listdir(data_dir) if file.endswith('_image.nii.gz')]
        self.label_files = [file for file in os.listdir(data_dir) if file.endswith('_label.nii.gz')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        label_file = image_file.replace('_image.nii.gz', '_label.nii.gz')
        if label_file in self.label_files:
            image_path = os.path.join(self.data_dir, image_file)
            label_path = os.path.join(self.data_dir, label_file)

            # Load image
            image_nii = nib.load(image_path)
            image_data = image_nii.get_fdata().astype(np.float32)

            # Normalize or preprocess image data as needed
            image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

            # Add channel dimension (assuming single channel data)
            image_data = np.expand_dims(image_data, axis=0)

            # Load label
            label_nii = nib.load(label_path)
            label_data = label_nii.get_fdata().astype(np.int32)
            # label_data = np.expand_dims(label_data, axis=0)

            return torch.tensor(image_data), torch.tensor(label_data, dtype=torch.long), image_file
        else:
            print(f'Label file not found for {image_file}')
            return None

if __name__ == '__main__':
    # Example usage:
    data_dir = '/home/lsy/PycharmProjects/VPTTA/mmwhs_processed/MR'
    dataset = MMWHS_MRDataset(data_dir)

    # Test the dataset
    for i in range(5):  # Load 5 random samples
        sample = dataset[i]
        volume, label = sample

