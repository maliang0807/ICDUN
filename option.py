import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--task', type=str, default='Dereflection', help='Dereflection')
parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
parser.add_argument("--scale_factor", type=int, default=1, help="4, 2")
parser.add_argument('--model_name', type=str, default='LF', help="model name")
parser.add_argument('--model_name1', type=str, default='LFRR', help="model name")

# Pretrained_model
parser.add_argument("--use_pre_ckpt", type=bool, default=True, help="use pre model ckpt")
# train
parser.add_argument("--resume", type=str,
                    default='',
                    help="path for retrain")
# test
parser.add_argument("--retrain", type=str,
                    default='',
                    help="path for test")

# Dataset
parser.add_argument('--path_for_train', type=str,
                    default='')
parser.add_argument('--path_for_test', type=str,
                    default='')
parser.add_argument('--data_name', type=str, default='ALL',
                    help='EPFL, HCI_new, HCI_old, INRIA_Lytro, Stanford_Gantry, ALL(of Five Datasets)')


# Trainingset
parser.add_argument('--path_log', type=str, default='')
parser.add_argument('--save_prefix', type=str, default='LFRR')

parser.add_argument('--patch_size', type=int, default=128, help="patch size")
parser.add_argument('--stride', type=int, default=4)
parser.add_argument('--n_patches_per_image', type=int, default=10,
                    help='n_patches_per_image')
parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
parser.add_argument('--n_steps', type=int, default=60, help='number of epochs to update learning rate')
parser.add_argument('--gamma', type=float, default=0.7, help='gamma')
parser.add_argument('--epoch', default=600, type=int, help='Epoch to run [default: 50]')

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_workers', type=int, default=8, help='num workers of the Data Loader')
parser.add_argument('--local_rank', dest='local_rank', type=int, default=0, )
parser.add_argument('--MGPU', type=int, default=2, help='num workers of the Data Loader')

parser.add_argument("--disp_record", type=bool, default=True, help="use pre model ckpt")
parser.add_argument('--iterations', type=int, default=4, help='Number of iterations for the LFRR network')
parser.add_argument('--fs', type=int, default=7, help='windows-size')


parser.add_argument('--channel', type=int, default=64, help='num workers of the Data Loader')
parser.add_argument('--layers', type=int, default=3, help='num workers of the Data Loader')


parser.add_argument('--k_nbr', default=6, type=int, help='')

args = parser.parse_args()

if args.task == 'Dereflection':
    args.angRes_in = args.angRes
    args.angRes_out = args.angRes
    args.patch_size_for_test = 32
    args.stride_for_test = 16
    args.minibatch_for_test = 1
