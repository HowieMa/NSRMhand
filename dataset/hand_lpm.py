import os
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt


parts = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]
]


groups6 = [
    [1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18, 19], [0, 4, 8, 12, 16],
]

groups1 = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
]


class HandDataset_LPM(Dataset):
    def __init__(self, data_root, mode='train', sigma=1, lsh_sigma=1, group='G1'):
        self.img_size = 368
        self.joints = 21  # 21 heat maps
        self.stride = 8
        self.label_size = self.img_size // self.stride  # 46
        self.sigma = sigma  # gaussian center heat map sigma
        self.lsh_sigma = lsh_sigma
        self.mode = mode
        if group == 'G6':
            self.group = groups6
            self.group_c = 6
        elif group == 'G1':
            self.group = groups1
            self.group_c = 1
        else:
            self.group = None
            self.group_c = 0

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
        img_label = self.all_labels[img_name]  # origin label list  21 * 2
        label = np.asarray(img_label)  # 21 * 2
        label[:, 0] = label[:, 0] * self.img_size / self.stride / w
        label[:, 1] = label[:, 1] * self.img_size / self.stride / h

        label_maps = self.gen_label_heatmap(label)

        # ******************** get Limb Segment map ********************
        lsh_maps = self.generate_lpm(label)
        if self.group:
            ori_maps = lsh_maps
            lsh_maps = self.limb_group(ori_maps, self.group_c, self.group)  # get group
            if self.group_c > 1:
                g1_maps = self.limb_group(ori_maps, 1, groups1)  # size:(1,46,46)
                lsh_maps = np.concatenate([g1_maps, lsh_maps], axis=0)

        lsh_maps = torch.from_numpy(lsh_maps).float()

        return image, label_maps, lsh_maps, img_name, w, h

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

    def generate_lpm(self, label):
        """
        get ridge heat map base on the distance to a line segment
        Formula basis: https://www.cnblogs.com/flyinggod/p/9359534.html
        """
        limb_maps = np.zeros((20, self.label_size, self.label_size))
        x, y = np.meshgrid(np.arange(self.label_size), np.arange(self.label_size))
        count = 0
        for part in parts:              # 20 parts
            x1, y1 = label[part[0]]        # vector start
            x2, y2 = label[part[1]]        # vector end

            cross = (x2 - x1) * (x - x1) + (y2 - y1) * (y - y1)
            length2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
            r = (cross + 1e-8) / (length2 + 1e-8)
            px = x1 + (x2 - x1) * r
            py = y1 + (y2 - y1) * r

            mask1 = cross <= 0              # 46 * 46
            mask2 = cross >= length2
            mask3 = 1 - mask1 | mask2

            D2 = np.zeros((self.label_size, self.label_size))
            D2 += mask1.astype('float32') * ((x - x1) * (x - x1) + (y - y1) * (y - y1))
            D2 += mask2.astype('float32') * ((x - x2) * (x - x2) + (y - y2) * (y - y2))
            D2 += mask3.astype('float32') * ((x - px) * (x - px) + (py - y) * (py - y))

            limb_maps[count] = np.exp(-D2 / 2.0 / self.lsh_sigma / self.lsh_sigma)  # numpy 2d
            count += 1
        return limb_maps

    def limb_group(self, limb_maps, groupc, modelgroup):
        # ************ Grouping Limb Maps ************
        ridegemap_group = np.zeros((groupc, self.label_size, self.label_size))
        count = 0
        for group in modelgroup:    # group6 or group1
            for g in group:
                group_tmp = ridegemap_group[count, :, :]
                limb_tmp = limb_maps[g, :, :]
                max_id = group_tmp < limb_tmp    #
                group_tmp[max_id] = limb_tmp[max_id]
                ridegemap_group[count, :, :] = group_tmp
            count += 1
        return ridegemap_group


# test case
if __name__ == "__main__":
    data_root = '../data_sample/cmuhand'

    print('G6 ============>')
    data = HandDataset_LPM(data_root=data_root, mode='train',group='G6')
    image, label_map, lsh_map, img_name, w, h = data[0]

    # ***************** draw Limb map *****************
    lab = np.asarray(lsh_map)
    group_out_labels = np.zeros((lab.shape[1], lab.shape[1] * lab.shape[0]))
    for i in range(lab.shape[0]):
        group_out_labels[:, lab.shape[1] * i:lab.shape[1] * i + lab.shape[1]] = lab[i, :, :]
    plt.imsave('lpm/lpm_g1_6.jpg', group_out_labels)

    print('G1 ============>')
    data = HandDataset_LPM(data_root=data_root, mode='train', group='G1')
    image, label_map, lsh_map, img_name, w, h = data[0]

    # ***************** draw Limb map *****************
    lab = np.asarray(lsh_map)
    group_out_labels = np.zeros((lab.shape[1], lab.shape[1] * lab.shape[0]))
    for i in range(lab.shape[0]):
        group_out_labels[:, lab.shape[1] * i:lab.shape[1] * i + lab.shape[1]] = lab[i, :, :]
    plt.imsave('lpm/lpm_g1.jpg', group_out_labels)
