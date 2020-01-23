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


class HandDataset_LDM(Dataset):
    def __init__(self, data_root, mode='train', sigma=1, width=1, group='G1'):
        self.img_size = 368
        self.joints = 21  # 21 heat maps
        self.stride = 8
        self.label_size = self.img_size // self.stride  # 46
        self.sigma = sigma  # gaussian center heat map sigma
        self.width = width
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

        # ******************** get Limb map ********************
        lam_maps = self.generate_ldm(label)
        if self.group:
            ori_maps = lam_maps
            lam_maps = self.limb_group(ori_maps, self.group_c, self.group)  # C,46,46
            if self.group_c > 1:
                g1_maps = self.limb_group(ori_maps, 1, groups1)         # size:(1,46,46)
                lam_maps = np.concatenate([g1_maps, lam_maps], axis=0)

        lam_maps = torch.from_numpy(lam_maps).float()

        return image, label_maps, lam_maps, img_name, w, h

    def gen_label_heatmap(self, label):
        label = torch.Tensor(label)

        grid = torch.zeros((self.label_size, self.label_size, 2))       # size:(46,46,2)
        grid[..., 0] = torch.Tensor(range(self.label_size)).unsqueeze(0)
        grid[..., 1] = torch.Tensor(range(self.label_size)).unsqueeze(1)
        grid = grid.unsqueeze(0)    # size:(1,46,46,2)
        labels = label.unsqueeze(-2).unsqueeze(-2)

        exponent = torch.sum((grid - labels)**2, dim=-1)    # size:(21,46,46)
        heatmaps = torch.exp(-exponent / 2.0 / self.sigma / self.sigma)
        return heatmaps

    def generate_ldm(self, label):
        """
        generate 20 part affinity from 21 labels
        reference: https://github.com/NiteshBharadwaj/part-affinity/blob/master/src/data_process/coco_process_utils.py
        :param label:  list 21 * 2
        :param label_size: 46
        :return:
        """
        limb_maps = np.zeros((20, self.label_size, self.label_size))
        x, y = np.meshgrid(np.arange(self.label_size), np.arange(self.label_size))

        count = 0
        for part in parts:              # 20 parts
            x1, y1 = label[part[0]]        # vector start
            x2, y2 = label[part[1]]        # vector end

            length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            # unit vector
            v1 = (x2 - x1)/(length + 1e-8)  # in case the length is zero, so add 1e-8
            v2 = (y2 - y1)/(length + 1e-8)

            dist_along_part = v1 * (x - x1) + v2 * (y - y1)
            dist_per_part = np.abs(v2 * (x - x1) + (-v1) * (y - y1))

            mask1 = dist_along_part >= 0
            mask2 = dist_along_part <= length
            mask3 = dist_per_part <= self.width
            mask = mask1 & mask2 & mask3

            limb_maps[count, :, :] = mask.astype('float32')
            count += 1
        return limb_maps

    def limb_group(self, limb_maps, groupc, model_group):
        # ************ Grouping Limb Maps by Finger ************
        limb_group = np.zeros((groupc, self.label_size, self.label_size))
        for count, group in enumerate(model_group):
            for g in group:
                index = np.nonzero(limb_maps[g, ...])
                limb_group[count, ...][index] = 1
        return limb_group


# test case
if __name__ == "__main__":
    data_root = '../data_sample/cmuhand'

    print('G6 ===========>')
    data = HandDataset_LDM(data_root=data_root, mode='train', group='G6')
    image, label_map, lsh_map, img_name, w, h = data[0]
    # ***************** draw Limb map *****************
    lab = np.asarray(lsh_map)
    group_out_labels = np.zeros((lab.shape[1], lab.shape[1] * lab.shape[0]))
    for i in range(lab.shape[0]):
        group_out_labels[:, lab.shape[1] * i:lab.shape[1] * i + lab.shape[1]] = lab[i, :, :]
    plt.imsave('ldm/ldm_g1_6.jpg', group_out_labels)

    print('G1 ===========>')
    data = HandDataset_LDM(data_root=data_root, mode='train', group='G1')
    image, label_map, lsh_map, img_name, w, h = data[0]
    # ***************** draw Limb map *****************
    lab = np.asarray(lsh_map)
    group_out_labels = np.zeros((lab.shape[1], lab.shape[1] * lab.shape[0]))
    for i in range(lab.shape[0]):
        group_out_labels[:, lab.shape[1] * i:lab.shape[1] * i + lab.shape[1]] = lab[i, :, :]
    plt.imsave('ldm/ldm_g1.jpg', group_out_labels)




