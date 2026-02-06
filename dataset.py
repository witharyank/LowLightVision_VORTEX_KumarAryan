import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UnderwaterDataset(Dataset):
    def __init__(self, input_dir, target_dir, augment=False):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.filenames = sorted(os.listdir(input_dir))
        self.augment = augment

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        input_path = os.path.join(self.input_dir, filename)
        target_path = os.path.join(self.target_dir, filename)

        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        if self.augment and torch.rand(1) > 0.5:
            input_img = transforms.functional.hflip(input_img)
            target_img = transforms.functional.hflip(target_img)

        input_img = self.transform(input_img)
        target_img = self.transform(target_img)

        return input_img, target_img
