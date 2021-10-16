# -*- coding:utf-8 -*-
import os
import numpy as np
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import Dataset, DataLoader


##################################################
# define dataloader class
##################################################
def default_loader(path):
    img_pil = Image.open(path)
    img_pil = img_pil.resize((224, 224))
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        # transforms.Scale(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    img_tensor = preprocess(img_pil)
    return img_tensor


class Trainset(Dataset):
    def __init__(self, file_train, number_train, loader=default_loader):
        self.images = file_train
        self.target = number_train
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img, target

    def __len__(self):
        return len(self.images)


def getDataset(path):
    image = []
    label = []
    label_n = []
    label_dict = {}
    for index, files in enumerate(os.listdir(path)):
        for images in os.listdir(os.path.join(path, files)):
            images_path = os.path.join(os.path.join(path, files), images)
            image.append(images_path)
            label.append(files)
            label_n.append(index)
        label_dict[index] = files
    # 输出当前的数据
    # print(len(image))
    # print(len(label))
    # 随机打乱数据集顺序
    image_len = len(image)
    index = np.arange(image_len)
    np.random.shuffle(index)
    return np.array(image)[index], np.array(label_n)[index], label_dict

