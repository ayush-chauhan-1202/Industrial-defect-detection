import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A

class DefectDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.images = image_paths
        self.masks = mask_paths

        self.aug = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks[idx], 0)

        augmented = self.aug(image=img, mask=mask)
        img = augmented["image"] / 255.0
        mask = augmented["mask"] / 255.0

        img = np.transpose(img, (2, 0, 1)).astype("float32")
        mask = np.expand_dims(mask, 0).astype("float32")

        return img, mask
