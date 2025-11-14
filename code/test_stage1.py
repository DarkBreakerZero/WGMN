import os

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

from config import opt
from utils import non_model, in_model
from make_dataset import test_Dataset
from net import model_MambaSR

import numpy as np

from tqdm import tqdm
import SimpleITK as sitk
import warnings
import argparse
warnings.filterwarnings("ignore")

import numpy as np
import torch
import SimpleITK as sitk
from torch.autograd import Variable


def test(**kwargs):

    kwargs, data_info_dict = non_model.read_kwargs(kwargs)
    opt.load_config('/home/lintong/MBSRN/configure/all.txt')
    config_dict = opt._spec(kwargs)


    save_output_folder = '/home/lintong/MBSRN/output/'
    os.makedirs(save_output_folder, exist_ok=True)

    model_path = '/home/lintong/MBSRN/models/HD_Mamba/1701_Loss_0.0070_PSNR_39.2799.pkl'
    save_dict = torch.load(model_path)
    config_dict = save_dict['config_dict']

    config_dict['mode'] = 'test'
    opt._spec(config_dict)

    data_gpu = opt.gpu_idx
    torch.cuda.set_device(data_gpu)

    load_net = save_dict['net']
    load_model_dict = load_net.state_dict()

    net = model_MambaSR.MBSRN3d_v1()
    net.load_state_dict(load_model_dict)

    net.cuda()
    net.eval()
    del save_dict

    test_list = data_info_dict['test']
    test_set = test_Dataset(test_list, opt.path_key)
    test_data_num = len(test_set.img_list)
    test_batch = Data.DataLoader(dataset=test_set, batch_size=opt.val_bs, shuffle=False,
                                 num_workers=opt.test_num_workers)
    print('load test data done, num =', test_data_num)
    print('input cube size:', opt.vc_z, opt.vc_y, opt.vc_x)

    psnr_list = []
    with torch.no_grad():
        for i, return_list in tqdm(enumerate(test_batch)):
            case_name, x, y, pos_list = return_list
            case_name = case_name[0]

            x = x.squeeze().data.numpy()
            y = y.squeeze().data.numpy()
            y_pre = np.zeros_like(y)

            print(x.shape)
            print(y.shape)

            pos_list = pos_list.data.numpy()[0]

            for pos_idx, pos in enumerate(pos_list):
                tmp_x = x[pos_idx]
                tmp_pos_z, tmp_pos_y, tmp_pos_x = pos

                tmp_x = torch.from_numpy(tmp_x)
                tmp_x = tmp_x.unsqueeze(0).unsqueeze(0)
                im = Variable(tmp_x.type(torch.FloatTensor).cuda())
                tmp_y_pre = net(im)
                tmp_y_pre = torch.clamp(tmp_y_pre, 0, 1)
                y_for_psnr = tmp_y_pre.data.squeeze().cpu().numpy()

                D = y_for_psnr.shape[0]
                pos_z_s = opt.ratio * tmp_pos_z + (opt.ratio - opt.ratio // 2)
                pos_y_s = tmp_pos_y
                pos_x_s = tmp_pos_x

                y_pre[pos_z_s: pos_z_s + D, pos_y_s:pos_y_s + opt.vc_y, pos_x_s:pos_x_s + opt.vc_x] = y_for_psnr

            del tmp_y_pre, im

            y_pre = np.array(y_pre, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            psnr = non_model.cal_psnr(y_pre[5:-5], y[5:-5])
            psnr_list.append(psnr)

            # y_pre = y_pre * (2048 + 1024) - 1024
            # y_pre = y_pre.astype('int16')

            save_name_pre = save_output_folder + case_name
            output_pre = sitk.GetImageFromArray(y_pre)
            sitk.WriteImage(output_pre, save_name_pre)

        psnr_test = np.array(psnr_list).mean()
        print('Mean PSNR: %.4f' % psnr_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script with command line args")

    # 添加你想支持的参数，这里只是示例，请根据你的实际需求添加更多
    parser.add_argument('--path_key', type=str, default='MSD')
    parser.add_argument('--net_idx', type=str, default='HD_Mamba')
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--models', type=str, default='mambav2')

    args = parser.parse_args()

    # 将 argparse.Namespace 转换为 dict
    kwargs = vars(args)

    # 调用 train 函数
    test(**kwargs)


