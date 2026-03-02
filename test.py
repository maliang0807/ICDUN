import importlib
import torch
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import MultiTestSetDataLoader
from collections import OrderedDict
from train import *
import imageio
import torch.nn as nn
import cv2
import einops
from scipy.io import savemat
import time
from utils.inference_method1 import test_mo as test_m1


def main(args):

    log_dir, _, result_dir = create_dir(args)
    result_dir = result_dir.joinpath('')
    result_dir.mkdir(exist_ok=True)


    logger = Logger(log_dir, args)


    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)


    logger.log_string('\nLoad Test Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    logger.log_string("The number of test data is: %d" % length_of_tests)


    logger.log_string('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name + '.' + args.model_name1
    MODEL = importlib.import_module(MODEL_PATH)

    net = MODEL.DeepUnfoldingNet(args)
    net = net.to(device)
    net = nn.DataParallel(net)
    cudnn.benchmark = True


    if args.retrain:
        retrain_path = args.retrain
        if os.path.isfile(retrain_path):
            logger.log_string("\n==> loading checkpoint {} for retrain".format(retrain_path))
            checkpoint = torch.load(retrain_path)
            net.load_state_dict(checkpoint['model'])
        else:
            logger.log_string("\n==> no model found at '{}'".format(retrain_path))
    else:
        logger.log_string("\n==> This is not a retrained experiment")


    logger.log_string('PARAMETER ...')
    logger.log_string(args)


    logger.log_string('\nStart test...')
    with torch.no_grad():

        psnr_testset = []
        ssim_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]

            save_dir = result_dir.joinpath(test_name)
            print(save_dir)
            save_dir.mkdir(exist_ok=True)


            psnr_iter_test, ssim_iter_test, LF_name = test_m1(test_loader, device, net, save_dir, logger=None, excel_file='Test_PSNR_SSIM.xlsx')


            psnr_epoch_test = float(np.array(psnr_iter_test).mean())
            psnr_testset.append(psnr_epoch_test)
            ssim_epoch_test = float(np.array(ssim_iter_test).mean())
            ssim_testset.append(ssim_epoch_test)
            logger.log_string('Test on %s, psnr/ssim is %.5f/%.5f' % (test_name, psnr_epoch_test, ssim_epoch_test))


        psnr_mean_test = float(np.array(psnr_testset).mean())
        ssim_mean_test = float(np.array(ssim_testset).mean())
        logger.log_string('The mean psnr on testsets is %.3f, mean ssim is %.4f' % (psnr_mean_test, ssim_mean_test))
    pass


if __name__ == '__main__':
    from option import args

    args.path_log = args.path_log + args.save_prefix
    main(args)
