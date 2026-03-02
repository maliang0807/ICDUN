import os

import einops
from torch.utils.data import Dataset
from skimage import metrics
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from utils import *
import einops
from torchvision.transforms import Resize

# def re_shape(img, h_re=576, w_re=384):
#     img = einops.rearrange(img, 'c (u h) (v w) -> c u h v w', u=5, v=5)
#     img_reshape = img[:,:,:h_re,:,:w_re]
#     img_reshape = einops.rearrange(img_reshape, 'c u h v w -> c (u h) (v w)')
#     return img_reshape

# class TrainSetDataLoader(Dataset):
#     def __init__(self, args):
#         super(TrainSetDataLoader, self).__init__()
#         self.angRes_in = args.angRes_in
#         self.angRes_out = args.angRes_out
#         if args.task == 'SR':
#             self.dataset_dir = args.path_for_train + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
#                                str(args.scale_factor) + 'x/'
#         elif args.task == 'RE':
#             self.dataset_dir = args.path_for_train + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
#                                str(args.angRes_out) + 'x' + str(args.angRes_out) + '/'
#             pass
#         elif args.task == 'Dereflection':
#             self.dataset_dir = args.path_for_train
#             pass

#         if args.data_name == 'ALL':
#             self.data_list = os.listdir(self.dataset_dir)
#         else:
#             self.data_list = [args.data_name]

#         self.file_list = []
#         for data_name in self.data_list:
#             print("data_name", data_name)
#             tmp_list = os.listdir(self.dataset_dir + data_name)

#             for index, _ in enumerate(tmp_list):
#                 tmp_list[index] = data_name + '/' + tmp_list[index]

#             self.file_list.extend(tmp_list)

#         self.item_num = len(self.file_list)

#         self.patch_size = args.patch_size

#     def __getitem__(self, index):
#         file_name = [self.dataset_dir + self.file_list[index]]
#         with h5py.File(file_name[0], 'r') as hf:
#             trans_LF = np.array(hf.get('trans_LF'))  # trans_LF
#             blended_LF = np.array(hf.get('blended_LF')) # Lr_SAI_y
#             # syn_reflection_LF = np.array(hf.get('Hr_SAI_y')) # Hr_SAI_y


#             trans_LF = torch.from_numpy(trans_LF)
#             blended_LF = torch.from_numpy(blended_LF)
#             # print("trans_LF1111111", trans_LF.shape)
#             # trans_LF1111111
#             # torch.Size([3, 3120, 2160])
#             # window_size = random.randrange(self.patch_size, self.patch_size*2, 8)
#             window_size = self.patch_size

#             '''crop'''
#             trans_LF = einops.rearrange(trans_LF, 'c (u H) (v W) -> c (u v) H W', u=self.angRes_in, v=self.angRes_in)
#             blended_LF = einops.rearrange(blended_LF, 'c (u H) (v W) -> c (u v) H W', u=self.angRes_in, v=self.angRes_in)
#             _, _, H, W = trans_LF.size()
#             x = random.randrange(0, H - window_size, 8)
#             y = random.randrange(0, W - window_size, 8)

#             trans_LF = trans_LF[:, :, x:x + window_size, y:y + window_size]  # [ah,aw,ph,pw]
#             blended_LF = blended_LF[:, :, x:x + window_size, y:y + window_size]  # [ah,aw,ph,pw]

#             # torch_resize = Resize([self.patch_size,self.patch_size]) # 定义Resize类对象
#             # trans_LF = torch_resize(trans_LF)
#             # blended_LF = torch_resize(blended_LF)

#             trans_LF = einops.rearrange(trans_LF, 'c (u v) H W -> c (u H) (v W)', u=self.angRes_in, v=self.angRes_in)
#             blended_LF = einops.rearrange(blended_LF, 'c (u v) H W -> c (u H) (v W)', u=self.angRes_in,
#                                           v=self.angRes_in)

#             '''augmentation'''
#             trans_LF, blended_LF = augmentation(trans_LF, blended_LF)
#             # print("trans_LF", trans_LF.shape)
#             # trans_LF
#             # torch.Size([3, 480, 480])
#             # reflection_LF = blended_LF - trans_LF
#             # trans_LF = ToTensor()(trans_LF.copy())
#             # blended_LF = ToTensor()(blended_LF.copy())
#             # reflection_LF = ToTensor()(syn_reflection_LF.copy())

#         Lr_angRes_in = self.angRes_in
#         Lr_angRes_out = self.angRes_out

#         return  blended_LF, trans_LF, [Lr_angRes_in, Lr_angRes_out]

#     def __len__(self):
#         return self.item_num
# class TrainSetDataLoader(Dataset):
#     def __init__(self, args):
#         super(TrainSetDataLoader, self).__init__()
#         self.angRes_in = args.angRes_in
#         self.angRes_out = args.angRes_out
#         self.patch_size = args.patch_size
#         self.stride = args.stride if hasattr(args, 'stride') else self.patch_size // 2  # 默认 50% overlap

#         if args.task == 'SR':
#             self.dataset_dir = os.path.join(args.path_for_train, f'SR_{args.angRes_in}x{args.angRes_in}_{args.scale_factor}x/')
#         elif args.task == 'RE':
#             self.dataset_dir = os.path.join(args.path_for_train, f'RE_{args.angRes_in}x{args.angRes_in}_{args.angRes_out}x{args.angRes_out}/')
#         elif args.task == 'Dereflection':
#             self.dataset_dir = args.path_for_train

#         if args.data_name == 'ALL':
#             self.data_list = os.listdir(self.dataset_dir)
#         else:
#             self.data_list = [args.data_name]

#         self.file_list = []
#         self.patch_index = []  # (file_id, x, y)

#         for file_id, data_name in enumerate(self.data_list):
#             print("data_name", data_name)
#             tmp_list = os.listdir(os.path.join(self.dataset_dir, data_name))
#             for subname in tmp_list:
#                 file_path = os.path.join(data_name, subname)
#                 full_path = os.path.join(self.dataset_dir, file_path)
#                 with h5py.File(full_path, 'r') as hf:
#                     trans = np.array(hf.get('trans_LF'))
#                     c, h, w = trans.shape
#                     H = h // self.angRes_in
#                     W = w // self.angRes_in
#                     for x in range(0, H - self.patch_size + 1, self.stride):
#                         for y in range(0, W - self.patch_size + 1, self.stride):
#                             self.patch_index.append((file_path, x, y))
#                 self.file_list.append(file_path)

#     def __len__(self):
#         return len(self.patch_index)

#     def __getitem__(self, index):
#         file_path, x, y = self.patch_index[index]
#         full_path = os.path.join(self.dataset_dir, file_path)

#         with h5py.File(full_path, 'r') as hf:
#             trans_LF = torch.from_numpy(np.array(hf.get('trans_LF')))
#             blended_LF = torch.from_numpy(np.array(hf.get('blended_LF')))

#         trans_LF = einops.rearrange(trans_LF, 'c (u H) (v W) -> c (u v) H W', u=self.angRes_in, v=self.angRes_in)
#         blended_LF = einops.rearrange(blended_LF, 'c (u H) (v W) -> c (u v) H W', u=self.angRes_in, v=self.angRes_in)

#         trans_LF = trans_LF[:, :, x:x + self.patch_size, y:y + self.patch_size]
#         blended_LF = blended_LF[:, :, x:x + self.patch_size, y:y + self.patch_size]

#         trans_LF = einops.rearrange(trans_LF, 'c (u v) H W -> c (u H) (v W)', u=self.angRes_in, v=self.angRes_in)
#         blended_LF = einops.rearrange(blended_LF, 'c (u v) H W -> c (u H) (v W)', u=self.angRes_in, v=self.angRes_in)

#         trans_LF, blended_LF = augmentation(trans_LF, blended_LF)

#         return blended_LF, trans_LF, [self.angRes_in, self.angRes_out]

# class TrainSetDataLoader(Dataset):
#     def __init__(self, args):
#         super(TrainSetDataLoader, self).__init__()
#         self.angRes_in = args.angRes_in
#         self.angRes_out = args.angRes_out
#         self.patch_size = args.patch_size
#         self.stride = args.stride
#         self.n_patches_per_image = args.n_patches_per_image  # 每图随机 patch 数

#         # 数据路径设置
#         if args.task == 'SR':
#             self.dataset_dir = os.path.join(args.path_for_train, f'SR_{args.angRes_in}x{args.angRes_in}_{args.scale_factor}x')
#         elif args.task == 'RE':
#             self.dataset_dir = os.path.join(args.path_for_train, f'RE_{args.angRes_in}x{args.angRes_in}_{args.angRes_out}x{args.angRes_out}')
#         elif args.task == 'Dereflection':
#             self.dataset_dir = args.path_for_train
#         else:
#             raise ValueError("Unsupported task type")

#         # 读取图像列表
#         self.data_list = os.listdir(self.dataset_dir) if args.data_name == 'ALL' else [args.data_name]

#         self.patch_index = []  # [(file_path, x, y)]

#         # 构建每张图像的 patch 坐标索引
#         for data_name in self.data_list:
#             data_folder = os.path.join(self.dataset_dir, data_name)
#             for fname in os.listdir(data_folder):
#                 file_path = os.path.join(data_name, fname)
#                 full_path = os.path.join(self.dataset_dir, file_path)

#                 with h5py.File(full_path, 'r') as hf:
#                     trans = np.array(hf.get('trans_LF'))
#                     c, h, w = trans.shape
#                     H = h // self.angRes_in
#                     W = w // self.angRes_in

#                     coords = [(x, y) for x in range(0, H - self.patch_size + 1, self.stride)
#                                        for y in range(0, W - self.patch_size + 1, self.stride)]

#                     selected = random.sample(coords, min(self.n_patches_per_image, len(coords)))
#                     for (x, y) in selected:
#                         self.patch_index.append((file_path, x, y))

#     def __len__(self):
#         return len(self.patch_index)

#     def __getitem__(self, index):
#         file_name, x, y = self.patch_index[index]
#         full_path = os.path.join(self.dataset_dir, file_name)

#         with h5py.File(full_path, 'r') as hf:
#             trans_LF = torch.from_numpy(np.array(hf.get('trans_LF')))
#             blended_LF = torch.from_numpy(np.array(hf.get('blended_LF')))

#         trans_LF = einops.rearrange(trans_LF, 'c (u H) (v W) -> c (u v) H W', u=self.angRes_in, v=self.angRes_in)
#         blended_LF = einops.rearrange(blended_LF, 'c (u H) (v W) -> c (u v) H W', u=self.angRes_in, v=self.angRes_in)

#         trans_LF = trans_LF[:, :, x:x + self.patch_size, y:y + self.patch_size]
#         blended_LF = blended_LF[:, :, x:x + self.patch_size, y:y + self.patch_size]

#         trans_LF = einops.rearrange(trans_LF, 'c (u v) H W -> c (u H) (v W)', u=self.angRes_in, v=self.angRes_in)
#         blended_LF = einops.rearrange(blended_LF, 'c (u v) H W -> c (u H) (v W)', u=self.angRes_in, v=self.angRes_in)

#         # 数据增强
#         trans_LF, blended_LF = augmentation(trans_LF, blended_LF)

#         return blended_LF, trans_LF, [self.angRes_in, self.angRes_out]


class TrainSetDataLoader(Dataset):
    def __init__(self, args):
        super(TrainSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        self.patch_size = args.patch_size
        self.n_patches_per_image = args.n_patches_per_image  # 每图随机 patch 数

        # 数据路径设置
        if args.task == 'SR':
            self.dataset_dir = os.path.join(args.path_for_train, f'SR_{args.angRes_in}x{args.angRes_in}_{args.scale_factor}x')
        elif args.task == 'RE':
            self.dataset_dir = os.path.join(args.path_for_train, f'RE_{args.angRes_in}x{args.angRes_in}_{args.angRes_out}x{args.angRes_out}')
        elif args.task == 'Dereflection':
            self.dataset_dir = args.path_for_train
        else:
            raise ValueError("Unsupported task type")

        # 读取图像列表
        self.data_list = os.listdir(self.dataset_dir) if args.data_name == 'ALL' else [args.data_name]

        self.file_paths = []
        for data_name in self.data_list:
            data_folder = os.path.join(self.dataset_dir, data_name)
            for fname in os.listdir(data_folder):
                file_path = os.path.join(data_name, fname)
                full_path = os.path.join(self.dataset_dir, file_path)
                self.file_paths.append(full_path)

    def __len__(self):
        return len(self.file_paths) * self.n_patches_per_image

    def __getitem__(self, index):
        # 计算图像索引和 patch 索引
        img_idx = index // self.n_patches_per_image
        file_path = self.file_paths[img_idx]

        with h5py.File(file_path, 'r') as hf:
            trans_LF = torch.from_numpy(np.array(hf.get('trans_LF')))
            blended_LF = torch.from_numpy(np.array(hf.get('blended_LF')))

        c, h, w = trans_LF.shape
        H = h // self.angRes_in
        W = w // self.angRes_in

        # 随机选择 patch 起点
        x = random.randint(0, H - self.patch_size)
        y = random.randint(0, W - self.patch_size)

        # 光场重排
        trans_LF = einops.rearrange(trans_LF, 'c (u H) (v W) -> c (u v) H W', u=self.angRes_in, v=self.angRes_in)
        blended_LF = einops.rearrange(blended_LF, 'c (u H) (v W) -> c (u v) H W', u=self.angRes_in, v=self.angRes_in)

        # 裁剪 patch
        trans_LF = trans_LF[:, :, x:x + self.patch_size, y:y + self.patch_size]
        blended_LF = blended_LF[:, :, x:x + self.patch_size, y:y + self.patch_size]

        # 光场恢复原始排列
        trans_LF = einops.rearrange(trans_LF, 'c (u v) H W -> c (u H) (v W)', u=self.angRes_in, v=self.angRes_in)
        blended_LF = einops.rearrange(blended_LF, 'c (u v) H W -> c (u H) (v W)', u=self.angRes_in, v=self.angRes_in)

        # 数据增强
        trans_LF, blended_LF = augmentation(trans_LF, blended_LF)

        return blended_LF, trans_LF, [self.angRes_in, self.angRes_out]

        
def MultiTestSetDataLoader(args):
    # get testdataloader of every test dataset
    data_list = None

    if args.data_name in ['ALL', 'RE_Lytro', 'RE_HCI']:
        if args.task == 'SR':
            dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                          str(args.scale_factor) + 'x/'
            data_list = os.listdir(dataset_dir)
        elif args.task == 'RE':
            dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                          str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name
            data_list = os.listdir(dataset_dir)
        elif args.task == 'Dereflection':
            dataset_dir = args.path_for_test
            data_list = os.listdir(dataset_dir)
    else:
        data_list = [args.data_name]

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name, Lr_Info=data_list.index(data_name))
        length_of_tests += len(test_Dataset)

        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=args.num_workers, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name = 'ALL', Lr_Info=None):
        super(TestSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        if args.task == 'SR':
            self.dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.scale_factor) + 'x/'
            self.data_list = [data_name]
        elif args.task == 'RE':
            self.dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name + '/'
            self.data_list = [data_name]
        elif args.task == 'Dereflection':
            self.dataset_dir = args.path_for_test
            self.data_list = [data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            trans_LF = np.array(hf.get('trans_LF'))  # trans_LF
            blended_LF = np.array(hf.get('blended_LF')) # Lr_SAI_y
            # syn_reflection_LF = np.array(hf.get('Hr_SAI_y')) # Hr_SAI_y
            trans_LF = torch.from_numpy(trans_LF)
            blended_LF = torch.from_numpy(blended_LF)

            # trans_LF = re_shape(trans_LF)
            # blended_LF = re_shape(blended_LF)

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out
        LF_name = self.file_list[index].split('/')[-1].split('.')[0]

        return  blended_LF, trans_LF, [Lr_angRes_in, Lr_angRes_out], LF_name

    def __len__(self):
        return self.item_num


def flip_SAI(data, angRes):
    if len(data.shape)==2:
        H, W = data.shape
        data = data.reshape(H, W, 1)

    H, W, C = data.shape
    data = data.reshape(angRes, H//angRes, angRes, W//angRes, C) # [U, H, V, W, C]
    data = data[::-1, ::-1, ::-1, ::-1, :]
    data = data.reshape(H, W, C)

    return data


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = torch.flip(data, dims=[2])
        label = torch.flip(label, dims=[2])
        # data = data[:, :, ::-1]
        # label = label[:, :, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = torch.flip(data, dims=[1])
        label = torch.flip(label, dims=[1])
        # data = data[:, ::-1, :]
        # label = label[:, ::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.permute(0, 2, 1)
        label = label.permute(0, 2, 1)
    return data, label

