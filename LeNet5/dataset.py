import os

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2

class POCDataset(Dataset):
    def __init__(self, root_dir, data_type="Training", resize=(224, 224), is_augment=False, transform=None, target_transform=None):
        super().__init__()
        self.resize = resize
        # self.is_augment = is_augment

        self.data_dir = os.path.join(root_dir, data_type)
        self.data_type = data_type

        self.image_names, self.labels = self.__process_data()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_names)

    def __preprocess_data(self, image):

        if self.transform:
            image = self.transform(np.array(image))

        return image

    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]

        label_map = {0: 'Chorionic_villi', 1:'Decidual_tissue', 2: 'Hemorrhage', 3: 'Trophoblastic_tissue'}

        image_path = os.path.join(self.data_dir, label_map[label], image_name)
        image = Image.open(image_path).convert('RGB')

        preprocessed = self.__preprocess_data(image)

        return preprocessed, torch.tensor(label).long()

    def __process_data(self):

        chorionic_villi = os.listdir(os.path.join(self.data_dir, 'Chorionic_villi'))
        decidual_tissue = os.listdir(os.path.join(self.data_dir, 'Decidual_tissue'))
        hemorrhage = os.listdir(os.path.join(self.data_dir, 'Hemorrhage'))
        trophoblastic = os.listdir(os.path.join(self.data_dir, 'Trophoblastic_tissue'))

        combined_images = chorionic_villi + decidual_tissue + hemorrhage + trophoblastic
        labels = [0] * len(chorionic_villi) + [1] * len(decidual_tissue) + [2] * len(hemorrhage) + [3] * len(trophoblastic)

        return combined_images, labels





