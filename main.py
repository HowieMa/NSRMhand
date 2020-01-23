import sys
import os
currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
sys.path.append(parentUrl)

import shutil
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.cpm_limb import CPMHandLimb
import dataset
from src.utils import set_logger, update_lr, get_pck_with_sigma, get_pred_coordinates, save_images, save_limb_images
from src import loss

# ***********************  Parameter  ***********************

parser = argparse.ArgumentParser()
parser.add_argument('config_file', help='config file for the experiment')
args = parser.parse_args()
configs = json.load(open('configs/' + args.config_file))

target_sigma_list = [0.04, 0.06, 0.08, 0.1, 0.12]
select_sigma = 0.1

model_name = 'EXP_' + configs["name"]
save_dir = os.path.join(model_name, 'checkpoint/')
test_pck_dir = os.path.join(model_name, 'test')

os.makedirs(save_dir, exist_ok=True)
os.makedirs(test_pck_dir, exist_ok=True)

shutil.copy('configs/' + args.config_file, model_name)

# training parameters ****************************
data_root = configs["data_root"]
learning_rate = configs["learning_rate"]
lr_decay_epoch = configs["lr_decay_epoch"]
batch_size = configs["batch_size"]
epochs = configs["epochs"]

weight_decay_epo = configs["weight_decay"]
weight_decay_ratio = configs["w_decay_ratio"]

weight_init = configs["weight"]
if "weight_g61" in configs:
    weight_g61 = configs["weight_g61"]
else:
    weight_g61 = 0.0

# data parameters ****************************
lshc = configs["limbc"]
group = configs["group"]

device_ids = configs["device"]      # multi-GPU
torch.cuda.set_device(device_ids[0])
cuda = torch.cuda.is_available()

logger = set_logger(os.path.join(model_name, 'train.log'))
logger.info("************** Experiment Name: {} **************".format(model_name))

# ******************** build model ********************
logger.info("Create Model ...")

model = CPMHandLimb(outc=21, lshc=lshc, pretrained=True)
if cuda:
    model = model.cuda(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids)

# ******************** data preparation  ********************
my_dataset = getattr(dataset, configs["dataset"])
train_data = my_dataset(data_root=data_root, mode='train', group=group)
valid_data = my_dataset(data_root=data_root, mode='valid', group=group)
test_data = my_dataset(data_root=data_root, mode='test', group=group)
logger.info('Total images in training data is {}'.format(len(train_data)))
logger.info('Total images in validation data is {}'.format(len(valid_data)))
logger.info('Total images in testing data is {}'.format(len(test_data)))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# ********************  ********************
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)


def train():
    logger.info('\nStart training ===========================================>')
    best_epo = -1
    max_pck = -1
    cur_lr = learning_rate
    cur_weight = weight_init

    logger.info('Initial learning Rate: {}'.format(learning_rate))

    for epoch in range(1, epochs + 1):
        logger.info('Epoch[{}/{}] ==============>'.format(epoch, epochs))
        model.train()
        train_cm_loss = []
        train_lm_loss = []

        # *************** Limb weight decay ***************
        # Since our limb representation is an intermediate product,
        # we don't need to get a very precise result after several epochs
        # Thus we decay the weight of limb loss after several epochs

        if weight_decay_epo > 0 and epoch % weight_decay_epo == 0:
            cur_weight = cur_weight * weight_decay_ratio
            logger.info('[Weight Decay] Current Weight [{}]'.format(cur_weight))

        for step, (img, cm_target, limb_target, img_name, w, h) in enumerate(train_loader):
            # *************** target prepare ***************
            limb_target = torch.stack([limb_target] * 3, dim=1)  # size:(bz,3,C,46,46)
            cm_target = torch.stack([cm_target] * 3, dim=1)      # size:(bz,3,21,46,46)
            if cuda:
                img = img.cuda()
                limb_target = limb_target.cuda()
                cm_target = cm_target.cuda()

            optimizer.zero_grad()
            limb_pred, cm_pred = model(img)
            # limb_pred (FloatTensor.cuda) size:(bz,3,C,46,46)  after sigmoid
            # cm_pred   (FloatTensor.cuda) size:(bz,3,21,46,46)

            # *************** calculate loss ***************
            if lshc == 1:       # For G1 only
                limb_loss = loss.ce_loss(limb_pred, limb_target)
            else:               # For G1 & 6
                g1_loss = loss.ce_loss(limb_pred[:, :, 0, ...], limb_target[:, :, 0, ...])
                g6_loss = loss.ce_loss(limb_pred[:, :, 1:, ...], limb_target[:, :, 1:, ...])
                limb_loss = g1_loss + weight_g61 * g6_loss

            cm_loss = loss.sum_mse_loss(cm_pred, cm_target)     # keypoint confidence map loss
            total_loss = cur_weight * limb_loss + cm_loss

            total_loss.backward()
            optimizer.step()

            train_cm_loss.append(cm_loss.item())
            train_lm_loss.append(limb_loss.item())

            if step % 50 == 0:
                logger.info('STEP: {}  LM LOSS {}'.format(step, limb_loss.item()))
                logger.info('          CM LOSS {}'.format(cm_loss.item()))

        # *************** save sample image after one epoch ***************
        save_images(cm_target[:, -1, ...].cpu(), cm_pred[:, -1, ...].cpu(),
                    epoch, img_name, save_dir)

        save_limb_images(limb_target[:, -1, ...].cpu(), limb_pred[:, -1, ...].cpu(),
                    epoch, img_name, save_dir)

        # *************** eval model after one epoch ***************
        eval_loss, cur_pck = eval(epoch, mode='valid')
        logger.info('EPOCH {} VALID PCK  {}'.format(epoch, cur_pck))
        logger.info('EPOCH {} TRAIN_CM_LOSS {}'.format(epoch, sum(train_cm_loss) / len(train_cm_loss)))
        logger.info('EPOCH {} TRAIN_LM_LOSS {}'.format(epoch, sum(train_lm_loss) / len(train_lm_loss)))
        logger.info('EPOCH {} VALID_LOSS {}'.format(epoch, eval_loss))

        # *************** save current model and best model ***************
        if cur_pck > max_pck:
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            best_epo = epoch
            max_pck = cur_pck
        logger.info('Current Best EPOCH is : {}, PCK is : {}\n**************\n'.format(best_epo, max_pck))

        # save current model
        torch.save(model.state_dict(), os.path.join(save_dir, 'final_epoch.pth'))

        # *************** update learning rate ***************
        if epoch % lr_decay_epoch == 0:
            logger.info("Learning Rate Decay ...")
            cur_lr /= 2
            update_lr(optimizer, cur_lr)

    logger.info('Train Done! ')
    logger.info('Best epoch is {}'.format(best_epo))
    logger.info('Best Valid PCK is {}'.format(max_pck))


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
        for step, (img, cm_target, limb_target, img_name, w, h) in enumerate(loader):
            if cuda:
                img = img.cuda()
            _, cm_pred = model(img)
            # limb_pred (FloatTensor.cuda) size:(bz,3,C,46,46)
            # cm_pred   (FloatTensor.cuda) size:(bz,3,21,46,46)

            all_pred_labels = get_pred_coordinates(cm_pred[:, -1, ...].cpu(),
                                                      img_name, w, h, all_pred_labels)
            loss_final = loss.sum_mse_loss(cm_pred[:, -1, ...].cpu(), cm_target)
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

logger.info('\nTESTING ============================>')
logger.info('Load Trained model !!!')
state_dict = torch.load(os.path.join(save_dir, 'best_model.pth'))
model.load_state_dict(state_dict)
eval(0, mode='test')

logger.info('Done!')




