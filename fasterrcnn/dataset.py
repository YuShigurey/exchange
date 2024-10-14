from torch.utils.data.dataset import Dataset
import os 
from PIL import Image
import torch
import json


class CustomDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annots = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        # Load images and annotations
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annot_path = os.path.join(self.root, "annotations", self.annots[idx])
        
        img = Image.open(img_path).convert("RGB")
        with open(annot_path) as f:
            annot = json.load(f)
        
        boxes = torch.as_tensor(annot['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(annot['labels'], dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        if self.transforms:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.imgs)