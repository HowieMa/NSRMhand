import argparse
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

from model.cpm_limb import CPMHandLimb
from PIL import Image, ImageDraw

cuda = torch.cuda.is_available()
device_id = [1]
torch.cuda.set_device(device_id[0])


def load_image(img_path):
    ori_im = Image.open(img_path)
    ori_w, ori_h = ori_im.size
    im = ori_im.resize((368, 368))
    image = transforms.ToTensor()(im)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])(image)  # (C,H,W)
    image = image.unsqueeze(0)  # (1,C,H,W)
    return ori_im, image, ori_w, ori_h


def get_image_coordinate(pred_map, ori_w, ori_h):

    """
    decode heatmap of one image to coordinates
    :param pred_map: Tensor  CPU     size:(1, 21, 46, 46)
    :return:
    label_list: Type:list, Length:21,  element: [x,y]
    """
    pred_map = pred_map.squeeze(0)
    label_list = []
    for k in range(21):
        tmp_pre = np.asarray(pred_map[k, :, :])  # 2D array  size:(46,46)
        corr = np.where(tmp_pre == np.max(tmp_pre))  # coordinate of keypoints in 46 * 46 scale

        # get coordinate of keypoints in origin image scale
        x = int(corr[1][0] * (int(ori_w) / 46.0))
        y = int(corr[0][0] * (int(ori_h) / 46.0))
        label_list.append([x, y])
    return label_list


def hand_pose_estimation(model, img_path='images/sample.jpg', save_path='images/sample_out.jpg'):
    with torch.no_grad():
        ori_im, img, ori_w, ori_h = load_image(img_path)
        if cuda:
            img = img.cuda()    # # Tensor size:(1,3,368,368)
        _, cm_pred = model(img)
        # limb_pred (FloatTensor.cuda) size:(bz,3,C,46,46)
        # cm_pred   (FloatTensor.cuda) size:(bz,3,21,46,46)

        coordinates = get_image_coordinate(cm_pred[:, -1].cpu(), ori_w, ori_h)
        # Type: list,   Length:21,      element:[x,y]
        ori_im = draw_point(coordinates, ori_im)
        print('save output to ', save_path)
        ori_im.save(save_path)
        return coordinates


def draw_point(points, im):
    i = 0
    draw = ImageDraw.Draw(im)

    for point in points:
        x = point[0]
        y = point[1]

        if i == 0:
            rootx = x
            rooty = y
        if i == 1 or i == 5 or i == 9 or i == 13 or i == 17:
            prex = rootx
            prey = rooty

        if i > 0 and i <= 4:
            draw.line((prex, prey, x, y), 'red')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'red', 'white')
        if i > 4 and i <= 8:
            draw.line((prex, prey, x, y), 'yellow')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'yellow', 'white')

        if i > 8 and i <= 12:
            draw.line((prex, prey, x, y), 'green')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'green', 'white')
        if i > 12 and i <= 16:
            draw.line((prex, prey, x, y), 'blue')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'blue', 'white')
        if i > 16 and i <= 20:
            draw.line((prex, prey, x, y), 'purple')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'purple', 'white')

        prex = x
        prey = y
        i = i + 1
    return im


if __name__ == "__main__":
    # ***********************  Parameter  ***********************

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default='weights/best_model.pth', help='trained model dir')
    parser.add_argument('--image_dir', default='images/', help='path for folder')
    args = parser.parse_args()

    # ******************** build model ********************
    # Limb Probabilistic Mask G1 & 6
    model = CPMHandLimb(outc=21, lshc=7, pretrained=False)
    if cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_id)

    state_dict = torch.load(args.resume)
    model.load_state_dict(state_dict)

    coordinate = hand_pose_estimation(model)
    print(coordinate)



