import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
from torch.utils.data.dataset import T_co


class FaceDataset(Dataset):
    def __init__(self, data_path, train_size=0.8, train=True):
        self.ages = []
        self.image_path = []
        self.image_label = []
        self.image_age = []
        self.train = train
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128))
            # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        with open(data_path, 'r') as f:
            for line in f.readlines():
                records = line.strip("\n").split("\t")
                if records[2] != 'u':
                    self.image_path.append(records[0])
                    self.image_label.append(records[2])
                    self.image_age.append(records[1])
        self.data_num = len(self.image_label)
        test_num = int(self.data_num * (1 - train_size))
        test_index = random.sample(range(len(self.image_label)), k=test_num)
        print("training at {} train samples and {} test samples".format(self.data_num - test_num, test_num))
        self.preprocess()
        self.test_path = [self.image_path[p] for p in test_index]
        self.test_label = [self.image_label[i] for i in test_index]
        self.test_age = [self.image_age[i] for i in test_index]
        self.image_path = [self.image_path[p] for p in range(self.data_num) if not test_index.__contains__(p)]
        self.image_label = [self.image_label[label] for label in range(self.data_num) if
                            not test_index.__contains__(label)]
        self.image_age = [self.image_age[a] for a in range(self.data_num) if not test_index.__contains__(a)]

    def preprocess(self):
        label = []
        for g in self.image_label:
            if g == 'f':
                label.append(0)
            elif g == 'm':
                label.append(1)
            else:
                label.append(2)
        self.image_label = label

    def __getitem__(self, index):
        if self.train:
            path = self.image_path[index]
            data = Image.open(path)
            data = self.transforms(data)
            return data, self.image_label[index]
        else:
            path = self.test_path[index]
            data = Image.open(path)
            data = self.transforms(data)
            return data, self.test_label[index]

    def __len__(self):
        if self.train:
            return len(self.image_label)
        else:
            return len(self.test_path)
