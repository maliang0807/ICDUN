import einops
from torch.utils.data import DataLoader
import torch.nn as nn
import importlib
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader
from collections import OrderedDict
import imageio
from skimage import metrics
from utils.func_pfm import *
import cv2
from tensorboardX import SummaryWriter
from utils.inference_method import test_m1
import os
from torchvision.utils import save_image
import pandas as pd  # 导入 pandas 库
import random
import time
import datetime





def main(args):

    log_dir, checkpoints_dir, val_dir = create_dir(args)


    logger = Logger(log_dir, args)


    writer = SummaryWriter(log_dir="{}logs/{}".format(log_dir, args.save_prefix),
                           comment="LFRRN 训练曲线")


    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)



    train_Dataset = TrainSetDataLoader(args)

    train_loader = DataLoader(dataset=train_Dataset, num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=True)



    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)



    MODEL_PATH = 'model.' + args.task + '.' + args.model_name + '.' + args.model_name1
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.DeepUnfoldingNet(args)
    net = net.to(device)
    cudnn.benchmark = True


    if args.MGPU == 4:
        net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])
        logger.log_string('\n使用4个GPU ...')
    elif args.MGPU == 2:
        net = nn.DataParallel(net, device_ids=[0, 1])
        logger.log_string('\n使用2个GPU ...')
    else:
        net = nn.DataParallel(net)

        



    logger.log_string(args)


    criterion = MODEL.get_loss(args).to(device)
    print(f"[Debug] args.lr = {args.lr}, type = {type(args.lr)}")
    print(f"[Debug] args.decay_rate = {args.decay_rate}, type = {type(args.decay_rate)}")


    optimizer = torch.optim.Adam(
        [paras for paras in net.parameters() if paras.requires_grad == True],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate,
        foreach=False
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)

    if args.resume:
        resume_path = args.resume
        if os.path.isfile(resume_path):
            logger.log_string("\n==> 加载检查点 {} 以恢复训练".format(resume_path))
            checkpoint = torch.load(resume_path)
            net.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
       

            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
        else:

            start_epoch = 0
    else:
        start_epoch = 0

    if args.retrain:
        retrain_path = args.retrain
        if os.path.isfile(retrain_path):
            logger.log_string("\n==> 加载检查点 {} 以重新训练".format(retrain_path))
            checkpoint = torch.load(retrain_path)
            net.load_state_dict(checkpoint['model'])
        else:
            logger.log_string("\n==> 在 '{}' 处未找到模型".format(retrain_path))
    else:
        logger.log_string("\n==> 这不是一个重新训练的实验")


    logger.log_string('\n开始训练...')

    for idx_epoch in range(start_epoch, args.epoch):
        epoch_start_time = time.time()  # === 开始计时 ===
        logger.log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))


        loss_epoch_train = train(train_loader, device, net, criterion, optimizer, idx_epoch + 1, writer)
        epoch_end_time = time.time()
        epoch_time_min = (epoch_end_time - epoch_start_time) / 60.0
   
        print('[{}] ==>第 {} 轮训练耗时: {:.2f} 分钟'.format(
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), idx_epoch + 1, epoch_time_min))



        logger.log_string('第 %d 轮训练, 损失是: %.5f' % (idx_epoch + 1, loss_epoch_train))
        writer.add_scalar("train/recon_loss", loss_epoch_train, idx_epoch + 1)



        if args.local_rank == 0 and (idx_epoch + 1) % 10 == 0 and idx_epoch > 50:
            save_ckpt_path = str(checkpoints_dir) + '/%s_%dx%d_epoch_%02d_model.pth' % (
                args.model_name, args.angRes_in, args.angRes_in, idx_epoch + 1)
            state = {
                'epoch': idx_epoch + 1,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(state, save_ckpt_path)
            logger.log_string('在 %s 保存第 %02d 轮模型' % (save_ckpt_path, idx_epoch + 1))


        scheduler.step()



def crop_center_3x3(data, angRes=5):

    B, C, H, W = data.shape
    h = H // angRes
    w = W // angRes
    start_u = angRes // 2 - 1
    end_u = start_u + 3

    out = []
    for u in range(start_u, end_u):
        row = []
        for v in range(start_u, end_u):
            patch = data[:, :, u*h:(u+1)*h, v*w:(v+1)*w]
            row.append(patch)
        out.append(torch.cat(row, dim=-1))
    out = torch.cat(out, dim=-2)
    return out

def train(train_loader, device, net, criterion, optimizer, epoch, writer):

    loss_iter_train = []
    for idx_iter, (data, label, data_info) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()
        angRes = 5

        data = data.to(device)


        label = label.to(device)
        data_sv = crop_center_3x3(data, angRes=5)

        label_sv = crop_center_3x3(label, angRes=5)

        optimizer.zero_grad()

        Disparity, out = net(data_sv, data_info)

        loss = criterion(Disparity, out, label_sv, data_info)

        loss.backward()


        optimizer.step()

        torch.cuda.empty_cache()

        loss_iter_train.append(loss.data.cpu())

        if idx_iter % 1 == 0:
            print("{}: Epoch {}, [{}/{}]:  Loss损失: {:.10f}".format(2025, epoch, idx_iter, len(train_loader),
                                                                     loss.cpu().data))
            writer.add_scalar("train/recon_loss_iter", loss.cpu().data, idx_iter + (epoch - 1) * len(train_loader))

    loss_epoch_train = float(np.array(loss_iter_train).mean())
    return loss_epoch_train



if __name__ == '__main__':
    from option import args

    args.path_log = args.path_log + args.save_prefix
    main(args)
