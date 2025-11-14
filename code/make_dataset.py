import numpy as np
import os
from utils import in_model
from config import opt
import torch
class train_Dataset:
    def __init__(self, img_list):
        self.img_path = '/mnt/no3/lintong/MSD_data/'
        self.img_list = img_list
        return

    def __getitem__(self, idx):
        case_name = self.img_list[idx]

        # for x and y
        tmp_img, tmp_mask = in_model.get_train_img(self.img_path, case_name)

        # tmp_img = in_model.add_win(tmp_img)
        # tmp_mask = in_model.add_win(tmp_mask)

        img = tmp_img[np.newaxis].copy()
        mask = tmp_mask[np.newaxis].copy()
        img = torch.tensor(img, dtype=torch.float32)
        mask =torch.tensor(mask, dtype=torch.float32)
        return_list = [img, mask]
        return return_list

    def __len__(self):
        return len(self.img_list)

class val_Dataset:
    def __init__(self, img_list):
        self.img_path = '/mnt/no3/lintong/MSD_data/'
        self.img_list = img_list
        return

    def __getitem__(self, idx):
        case_name = self.img_list[idx]

        # for x and y
        crop_img, pos_list, tmp_mask = in_model.get_val_img(self.img_path, case_name)
        crop_img = torch.tensor(crop_img, dtype=torch.float32)
        pos_list= torch.tensor(pos_list, dtype=torch.int64)
        # case_name = torch.tensor(case_name, dtype=torch.float32)
        tmp_mask = torch.tensor(tmp_mask, dtype=torch.float32)


        # crop_img = in_model.add_win(crop_img)
        # tmp_mask = in_model.add_win(tmp_mask)

        return_list = [case_name, crop_img, tmp_mask, pos_list]

        return return_list

    def __len__(self):
        return len(self.img_list)


class test_Dataset:
    def __init__(self, img_list, test_path_key):
        self.img_path = '/mnt/no3/lintong/MSD_data/'
        self.img_list = img_list
        self.test_path_key = test_path_key
        return

    def __getitem__(self, idx):
        case_name = self.img_list[idx]
        crop_img, pos_list, tmp_mask = in_model.get_test_img(self.img_path, case_name, self.test_path_key)
        crop_img = torch.tensor(crop_img, dtype=torch.float32)
        pos_list = torch.tensor(pos_list, dtype=torch.int64)
        # case_name = torch.tensor(case_name, dtype=torch.float32)
        tmp_mask = torch.tensor(tmp_mask, dtype=torch.float32)

        # crop_img = in_model.add_win(crop_img)
        # tmp_mask = in_model.add_win(tmp_mask)

        return_list = [case_name, crop_img, tmp_mask, pos_list]

        return return_list

    def __len__(self):
        return len(self.img_list)


