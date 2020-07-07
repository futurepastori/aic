import os
import json
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
from PIL import Image


TRANSFORM_IMAGENET = transforms.Compose([ 
    transforms.Resize(256),                         
    transforms.RandomCrop(224),                      
    transforms.RandomHorizontalFlip(),               
    transforms.ToTensor(),                           
    transforms.Normalize((0.485, 0.456, 0.406),      
                         (0.229, 0.224, 0.225))])


def get_loader(path: str, mode: str, batch_size: int, split_size=0.2) -> object:
    """
    Generate a PyTorch DataLoader based on a CustomDataset instance for
    a random subset of images

    :param path: root folder where all the banter is
    :param mode: either train or test, will determine __getitem__
    :param batch_size: self-explanatory
    :return: the DataLoader itself
    """
    #if mode == 'test':
    #    assert batch_size == 1, "Batch size must be 1"

    dataset = CustomDataset(path, mode, transform=TRANSFORM_IMAGENET)

    split_lengths = [int(np.floor(len(dataset) * (1-split_size))), 
                     int(np.ceil(len(dataset) * split_size))]

    train_split, test_split = random_split(dataset, split_lengths)

    data_loader = DataLoader(train_split if mode is "train" else test_split,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=1)

    return data_loader


class CustomDataset(Dataset):
    """
    Custom dataset class derived from PyTorch's Dataset for COCO
    with ImageNet standard transforms transforms

    :param path: root folder where all the banter is
    :param mode: either train or test, will determine __getitem__
    :param batch_size: self-explanatory
    """    
    def __init__(self, path, mode, transform):
        self.path = path
        self.mode = mode
        self.transform = transform
        with open(os.path.join(path, 'json', 'captions_coco.json'), 'r') as j:
            self.captions = json.load(j)
    
    def __getitem__(self, i):
        item = self.captions[i]
        impath = os.path.join(self.path, item['split'], item['filename'])
        img = Image.open(impath).convert('RGB')
        
        if self.mode == 'train':
            img = self.transform(img)
            caption = torch.LongTensor(item['caption'])
            length = item['length']

            return img, caption, length
        else: 
            img_trans = self.transform(img)
            caption = torch.LongTensor(item['caption'])
            length = item['length']
            photoid = int(item['filename'][15:-4])

            return np.array(img), img_trans, caption, length, photoid

    def __len__(self):
        return len(self.captions)