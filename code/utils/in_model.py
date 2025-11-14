import numpy as np
import SimpleITK as sitk
import cv2
import os
import random
from copy import deepcopy
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from config import opt

from builtins import range
import einops

path_dict = {
    'MSD': ['1mm', '5mm']
}
opt.load_config('/home/lintong/CTHNet-for-CT-Slice-Thickness-Reduction-main/configure/all.txt')
################################## for data ##################################
def get_train_img(img_path, case_name):
    path_thin, path_thick = path_dict[opt.path_key]

    case_mask_path = img_path + '%s/%s' % (path_thin, case_name)
    tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(case_mask_path))


    case_img_path = img_path + '%s/%s' % (path_thick, case_name)
    tmp_img = sitk.GetArrayFromImage(sitk.ReadImage(case_img_path))



    # random crop
    z = tmp_img.shape[0]
    z_s = random.randint(0, z - 1 - opt.c_z)
    y_s = random.randint(0, 512 - opt.c_y)
    x_s = random.randint(0, 512 - opt.c_x)
    z_e = z_s + opt.c_z
    y_e = y_s + opt.c_y
    x_e = x_s + opt.c_x

    crop_img = tmp_img[z_s:z_e, y_s:y_e, x_s:x_e]

    mask_z_s = z_s * 5 + 3
    mask_z_e = (z_e - 1) * 5 - 2

    # mask_z_s = z_s * 5
    # mask_z_e = (z_e - 1) * 5 +1

    crop_mask = tmp_mask[mask_z_s: mask_z_e, y_s:y_e, x_s:x_e]

    # H Flip
    if np.random.uniform() <= 0.2:
        crop_img = crop_img[:, :, ::-1]
        crop_mask = crop_mask[:, :, ::-1]

    return crop_img, crop_mask

def get_val_img(img_path, case_name):
    path_thin, path_thick = path_dict[opt.path_key]

    case_mask_path = img_path + '%s/%s' % (path_thin, case_name)
    tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(case_mask_path))

    case_img_path = img_path + '%s/%s' % (path_thick, case_name)
    tmp_img = sitk.GetArrayFromImage(sitk.ReadImage(case_img_path))

    if opt.mode != 'test':
        tmp_img = tmp_img[:, 128:-128, 128:-128]
        tmp_mask = tmp_mask[:, 128:-128, 128:-128]

    z, y, x = tmp_img.shape
    z_s = 0
    z_split = []
    while z_s + opt.vc_z < z:
        z_split.append(z_s)
        z_s += (opt.vc_z - 2)

    if z - opt.vc_z > z_split[-1]:
        z_split.append(z - opt.vc_z)

    y_split = np.arange(y // opt.vc_y) * opt.vc_y
    x_split = np.arange(x // opt.vc_x) * opt.vc_x

    crop_img = []
    pos_list = []

    for z_s in z_split:
        tmp_crop_img = deepcopy(tmp_img)[z_s:z_s + opt.vc_z]
        tmp_crop_img = einops.rearrange(tmp_crop_img, 'd (hn hs) (wn ws) -> (hn wn) d hs ws', hs=opt.vc_y, ws=opt.vc_x)
        crop_img.append(tmp_crop_img)
    crop_img = np.concatenate(crop_img, axis=0)

    for z_s in z_split:
        for y_s in y_split:
            for x_s in x_split:
                pos_list.append(np.array([z_s, y_s, x_s]))
    pos_list = np.array(pos_list)

    return crop_img, pos_list, tmp_mask

def get_test_img(img_path, case_name, path_key):
    path_thin, path_thick = path_dict[path_key]

    case_mask_path = img_path + '%s/%s' % (path_thin, case_name)
    tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(case_mask_path))

    case_img_path = img_path + '%s/%s' % (path_thick, case_name)
    tmp_img = sitk.GetArrayFromImage(sitk.ReadImage(case_img_path))

    z, y, x = tmp_img.shape
    z_s = 0
    z_split = []
    while z_s + opt.vc_z < z:
        z_split.append(z_s)
        z_s += (opt.vc_z - 3)

    if z - opt.vc_z > z_split[-1]:
        z_split.append(z - opt.vc_z)

    pad_y = max(0, 512 - y)
    pad_x = max(0, 512 - x)
    tmp_img = np.pad(tmp_img, ((0, 0), (0, pad_y), (0, pad_x)), mode='minimum')
    z, y, x = tmp_img.shape

    y_split = np.arange(y // opt.vc_y) * opt.vc_y
    x_split = np.arange(x // opt.vc_x) * opt.vc_x

    crop_img = []
    pos_list = []

    for z_s in z_split:
        tmp_crop_img = deepcopy(tmp_img)[z_s:z_s + opt.vc_z]
        tmp_crop_img = einops.rearrange(tmp_crop_img, 'd (hn hs) (wn ws) -> (hn wn) d hs ws', hs=opt.vc_y, ws=opt.vc_x)
        crop_img.append(tmp_crop_img)
    crop_img = np.concatenate(crop_img, axis=0)

    for z_s in z_split:
        for y_s in y_split:
            for x_s in x_split:
                pos_list.append(np.array([z_s, y_s, x_s]))

    pos_list = np.array(pos_list)

    return crop_img, pos_list, tmp_mask





################################## for pre-process ##################################
def add_win(img):
    left_win = -1024
    right_win = 2048
    img = img.clip(left_win, right_win)
    img = (img - left_win) / (right_win - left_win)
    return img


