import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from utils import config
from pathlib import Path


workers = int(config['data']['workers'])

path = Path(config['data']['path'])


class Dataset:

    tranform = transforms.Compose([
        transforms.Resize(eval(config['data']['img_size'])),
        transforms.ToTensor()
    ])

    @staticmethod
    def create():
        dataset = {}
        dataloader = {}

        for x in eval(config['data']['sets']):
            dataset[x] = datasets.ImageFolder(path/x, Dataset.tranform)
            dataloader[x] = DataLoader(
                dataset[x],
                batch_size=int(config['data']['batch']),
                num_workers=int(config['data']['workers']),
                shuffle=True)

        classes = dataset['train'].classes

        return dataset, dataloader, classes
