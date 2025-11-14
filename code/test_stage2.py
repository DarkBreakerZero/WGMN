import os
import scipy
import numpy as np
import torch.nn
# from datasets import dicom_file
from SR_dataset import SRDataset
from torch.utils.data import DataLoader
from torch.backends import cudnn
# from skimage.measure import compare_ssim as ssim
# from skimage.measure import compare_psnr as psnr
import time
import torch.optim as optim
import argparse
from tqdm import tqdm
from net.models import *
from utils.util import *
from datetime import datetime
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import os
import tifffile

def test(test_loader, g_model, output_dir):
    batch_time = AverageMeter()
    loss_mse_scalar = AverageMeter()
    loss_ssim_scalar = AverageMeter()
    end = time.time()

    g_model.eval()

    step = 0

    # 确保输出目录存在
    rdct_dir = os.path.join(output_dir, 'HRCT')
    ldct_dir = os.path.join(output_dir, 'SRCT')
    predict_dir = os.path.join(output_dir, 'Predict')
    os.makedirs(rdct_dir, exist_ok=True)
    os.makedirs(ldct_dir, exist_ok=True)
    os.makedirs(predict_dir, exist_ok=True)

    for batch_idx, data in enumerate(tqdm(test_loader)):
        RDCTImg = data["B"]
        RDCTImg = RDCTImg.cuda()

        LDCTImg = data["A"]
        LDCTImg = LDCTImg.cuda()

        with torch.no_grad():
            predictImg = g_model(LDCTImg)
            loss1 = loss_mse(predictImg, RDCTImg)
            loss2 = loss_ssim(predictImg, RDCTImg)

        loss_mse_scalar.update(loss1.item(), LDCTImg.size(0))
        loss_ssim_scalar.update(loss2.item(), LDCTImg.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # 假设你想保存每张图片
        for idx in range(RDCTImg.size(0)):
            # 转换为CPU并转换为numpy数组
            rdct_img = RDCTImg[idx].cpu().numpy()
            ldct_img = LDCTImg[idx].cpu().numpy()
            pred_img = predictImg[idx].cpu().numpy()

            # 保存图像
            tifffile.imsave(os.path.join(rdct_dir, f'batch_{batch_idx}_img_{idx}.tif'), rdct_img.squeeze())
            tifffile.imsave(os.path.join(ldct_dir, f'batch_{batch_idx}_img_{idx}.tif'), ldct_img.squeeze())
            tifffile.imsave(os.path.join(predict_dir, f'batch_{batch_idx}_img_{idx}.tif'), pred_img.squeeze())

        step += 1




if __name__ == "__main__":

    # opt = TrainOptions().parse()  # get training options
    # opt.dataroot = "H:/超分辨/noise"

    cudnn.benchmark = True

    #result_path = './runs/Ralph_Texture_Preserver_Holder/logs/'
    save_dir = '/models/'


    test_dataset = SRDataset("/mnt/no3/lintong/Texture_Preserver", "test")
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=6, shuffle=False)

    generator = UNet2d()
    discriminator = Discriminator2D()

    loss_mse = torch.nn.MSELoss()
    loss_ssim = SSIM()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0001)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=100, gamma=0.1)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=100, gamma=0.1)


    print("test")
    checkpoint_latest_g = torch.load(find_lastest_file(save_dir + 'G/'))
    generator = load_model_model_parallel(generator, checkpoint_latest_g)
    generator = generator.cuda()
    optimizer_g.load_state_dict(checkpoint_latest_g['optimizer_state_dict'])
    scheduler_g.load_state_dict(checkpoint_latest_g['lr_scheduler'])



    print('Latest checkpoint {0} loaded.'.format(find_lastest_file(save_dir)))

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    # log_dir = os.path.join(result_path, time_str)
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)

    output_dir = '/home/lintong/MBSRN/output_tp/'

    test(test_loader, generator, output_dir)

