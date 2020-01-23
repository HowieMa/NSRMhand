import os
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image
from src.augmentation import *
import imageio


class CMUHandDataset(Dataset):

    def __init__(self, data_root, mode='train', sigma=1):
        self.img_size = 368
        self.joints = 21  # 21 heat maps
        self.stride = 8
        self.label_size = self.img_size // self.stride  # 46
        self.sigma = sigma  # gaussian center heat map sigma
        self.mode = mode

        self.data_root = data_root
        self.img_names = json.load(open(os.path.join(self.data_root, 'partitions.json')))[mode]
        self.all_labels = json.load(open(os.path.join(self.data_root, 'labels.json')))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]  # '00000001.jpg'

        # ********************** get image **********************
        im = Image.open(os.path.join(self.data_root, 'imgs', img_name))
        w, h = im.size

        im = im.resize((self.img_size, self.img_size))

        image = transforms.ToTensor()(im)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(image)

        # ******************** get label map **********************
        img_label = self.all_labels[img_name]       # 21 * 2
        label = np.asarray(img_label)   # 21 * 2
        label[:, 0] = label[:, 0] * self.img_size / self.stride / w
        label[:, 1] = label[:, 1] * self.img_size / self.stride / h

        label_maps = self.gen_label_heatmap(label)

        return image, label_maps, img_name, w, h

    def gen_label_heatmap(self, label):
        label = torch.Tensor(label)

        grid = torch.zeros((self.label_size, self.label_size, 2))       # size:(46,46,2)
        grid[..., 0] = torch.Tensor(range(self.label_size)).unsqueeze(0)
        grid[..., 1] = torch.Tensor(range(self.label_size)).unsqueeze(1)
        grid = grid.unsqueeze(0)
        labels = label.unsqueeze(-2).unsqueeze(-2)
        exponent = torch.sum((grid - labels)**2, dim=-1)    # size:(21,46,46)
        heatmaps = torch.exp(-exponent / 2.0 / self.sigma / self.sigma)
        return heatmaps


# test case
if __name__ == "__main__":
    data_root = '../data_sample/cmuhand'

    data = CMUHandDataset(data_root=data_root, mode='train')

    image, label_maps, img_name, w, h = data[0]

    # ***************** draw label map *****************
    lab = np.asarray(label_maps)
    out_labels = np.zeros((46, 46))
    for i in range(21):
        out_labels += lab[i, :, :]
    imageio.imwrite('img/label.jpg', out_labels)
    # ***************** draw image *****************
    img = transforms.ToPILImage()(image)
    print(img.size)
    imageio.imwrite('img/img.jpg', img)


