from torch.utils.data import DataLoader
from dataset import CustomDataset
from transformations import get_transform

dataset = CustomDataset('path/to/dataset', transforms=get_transform(train=True))
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))