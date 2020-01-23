import sys
import os
currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
sys.path.append(parentUrl)

import numpy as np
import json
import scipy.misc as misc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.utils import *
from cpm import CPMHand
from cmuhand import CMUHandDataset


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


# ***********************  Parameter  ***********************
data_root = '/home/haoyum/data/CMUhand'
save_dir = 'ckpt/'
test_pck_dir = 'predict_test/'
make_dir(save_dir)
make_dir(test_pck_dir)


learning_rate = 1e-4
lr_decay_epoch = 40
batch_size = 32
epochs = 45
target_sigma_list = [ 0.04, 0.06, 0.08, 0.1, 0.12]
select_sigma = 0.1

# multi-GPU
device_ids = [0, 1]
torch.cuda.set_device(device_ids[0])
cuda = torch.cuda.is_available()

# ******************** build model ********************
model = CPMHand(21, pretrained=True)
if cuda:
    model = model.cuda(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids)

# ******************** data preparation  ********************
train_data = CMUHandDataset(data_root=data_root, mode='train')
valid_data = CMUHandDataset(data_root=data_root, mode='valid')
test_data = CMUHandDataset(data_root=data_root, mode='test')
print('Total images in training data is {}'.format(len(train_data)))
print('Total images in validation data is {}'.format(len(valid_data)))
print('Total images in testing data is {}'.format(len(test_data)))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# ******************** data preparation  ********************
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

def sum_mse_loss(pred, target):
    """
    :param pred:    Tensor  B,num_stage(3),num_channel(21),46,46
    :param target:
    :return:
    """
    criterion = nn.MSELoss(reduction='sum')
    loss = criterion(pred, target)
    return loss / (pred.shape[0] * 46.0 * 46.0)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    print('Start training ===========================================>')
    best_epo = -1
    max_pck = -1
    cur_lr = learning_rate
    print('Learning Rate: {}'.format(learning_rate))
    for epoch in range(1, epochs + 1):
        print('Epoch[{}/{}] ==============>'.format(epoch, epochs))
        model.train()
        train_loss = []

        for step, (img, label, img_name, w, h) in enumerate(train_loader):
            label = torch.stack([label] * 6, dim=1)  # bz * 6 * 21 * 46 * 46
            if cuda:
                img = img.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            pred_maps = model(img)   # (FloatTensor.cuda) size:(bz,6,21,46,46)
            loss = sum_mse_loss(pred_maps, label)  # total loss

            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print('STEP: {}  LOSS {}'.format(step, loss.item()))

            loss_final = sum_mse_loss(pred_maps[:, -1, ...].cpu(), label[:, -1, ...].cpu())
            train_loss.append(loss_final)

        # save sample image ****
        save_images(label[:, -1, ...].cpu(), pred_maps[:, -1, ...].cpu(),
                    epoch, img_name, save_dir)

        # eval model after one epoch
        eval_loss, cur_pck = eval(epoch, mode='valid')
        print('EPOCH {}  Valid PCK {}'.format(epoch, cur_pck))
        print('EPOCH {} TRAIN_LOSS {}'.format(epoch, sum(train_loss)/len(train_loss)))
        print('EPOCH {} VALID_LOSS {}'.format(epoch, eval_loss))

        if cur_pck > max_pck:
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            max_pck = cur_pck
            best_epo = epoch
        print('Current Best EPOCH is : {}\n**************\n'.format(best_epo))
        torch.save(model.state_dict(), os.path.join(save_dir, 'final_epoch.pth'))

        if epoch % lr_decay_epoch == 0:
            cur_lr /= 2
            update_lr(optimizer, cur_lr)

    print('Train Done!')
    print('Best epoch is {}'.format(best_epo))


def eval(epoch, mode='valid'):
    if mode is 'valid':
        loader = valid_loader
        gt_labels = valid_data.all_labels
    else:
        loader = test_loader
        gt_labels = test_data.all_labels

    with torch.no_grad():
        all_pred_labels = {}        # save predict results
        eval_loss = []
        model.eval()
        for step, (img, label, img_name, w, h) in enumerate(loader):
            if cuda:
                img = img.cuda()
            pred_maps = model(img)  # output 5D tensor:   bz * 6 * 21 * 46 * 46
            all_pred_labels = get_pred_coordinates(pred_maps[:, -1, ...].cpu(),
                                                      img_name, w, h, all_pred_labels)

            loss_final = sum_mse_loss(pred_maps[:, -1, ...].cpu(), label)
            eval_loss.append(loss_final)

        # ******** save predict labels for valid/test data ********
        if mode is 'valid':
            pred_save_dir = os.path.join(save_dir, 'e' + str(epoch) + '_val_pred.json')
        else:
            pred_save_dir = os.path.join(test_pck_dir, 'test_pred.json')
        json.dump(all_pred_labels, open(pred_save_dir, 'w'), sort_keys=True, indent=4)

        # ************* calculate and save PCKs  ************
        pck_dict = get_pck_with_sigma(all_pred_labels, gt_labels, target_sigma_list)

        if mode is 'valid':
            pck_save_dir = os.path.join(save_dir, 'e' + str(epoch) + '_pck.json')
        else:
            pck_save_dir = os.path.join(test_pck_dir, 'pck.json')
        json.dump(pck_dict, open(pck_save_dir, 'w'), sort_keys=True, indent=4)

        select_pck = pck_dict[select_sigma]
        eval_loss = sum(eval_loss)/len(eval_loss)
    return eval_loss, select_pck


train()

print('TESTING ============================>')
state_dict = torch.load(os.path.join(save_dir, 'best_model.pth'))
model.load_state_dict(state_dict)
eval(0, mode='test')


