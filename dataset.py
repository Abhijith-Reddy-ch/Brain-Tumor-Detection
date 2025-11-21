# dataset.py
import os
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 224

def get_transforms(img_size=IMG_SIZE):
    """
    Returns (train_transform, val_transform).
    Uses tuple/keyword forms to be compatible across albumentations versions.
    """
    size = (img_size, img_size)

    train_transform = A.Compose([
        # use Affine instead of ShiftScaleRotate (recommended)
        A.Affine(translate_percent=0.06, scale=(0.92, 1.08), rotate=15, p=0.8),
        A.RandomResizedCrop(size=size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        # CoarseDropout expects height/width ints; use a relative proportion
        A.CoarseDropout(max_holes=1, max_height=int(img_size * 0.08), max_width=int(img_size * 0.08), p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(),
        ToTensorV2(),
    ])
    return train_transform, val_transform

class BrainMRIDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        assert len(paths) == len(labels)
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = np.array(Image.open(p).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)['image']
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def make_file_lists(train_dir, test_dir):
    """
    Expects train_dir to have class subfolders and test_dir similarly.
    Returns: classes, train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
    """
    from sklearn.model_selection import train_test_split
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    class2idx = {c:i for i,c in enumerate(classes)}

    all_paths = []
    all_labels = []
    for c in classes:
        folder = os.path.join(train_dir, c)
        imgs = glob(os.path.join(folder, "*"))
        imgs = [p for p in imgs if p.lower().endswith(('.png','.jpg','.jpeg'))]
        all_paths += imgs
        all_labels += [class2idx[c]] * len(imgs)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.15, stratify=all_labels, random_state=42
    )

    test_paths = []
    test_labels = []
    for c in classes:
        folder = os.path.join(test_dir, c)
        imgs = glob(os.path.join(folder, "*"))
        imgs = [p for p in imgs if p.lower().endswith(('.png','.jpg','.jpeg'))]
        test_paths += imgs
        test_labels += [class2idx[c]] * len(imgs)

    return classes, train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
