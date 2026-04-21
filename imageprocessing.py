import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import torchvision.transforms as T



class ObjectDataset(Dataset):
    def __init__(
        self,
        img_dir="/content/Banner",
        label_dir="/content/obj_Train_data",
        max_objects=5
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.images = sorted(os.listdir(img_dir))
        self.max_objects = max_objects

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # 🔹 LOAD IMAGE (REAL IMAGE ENTERS HERE)
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # 🔹 LOAD LABELS
        label_path = os.path.join(
            self.label_dir,
            img_name.replace(".jpg", ".txt")
        )

        boxes = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    boxes.append(list(map(float, line.split())))

        boxes = torch.tensor(boxes) if boxes else torch.zeros((0, 5))

        # 🔹 PAD TO FIXED SIZE
        padded = torch.zeros((self.max_objects, 5))
        padded[:len(boxes)] = boxes[:self.max_objects]

        return image, padded
