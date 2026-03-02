import h5py
import numpy as np
import torch
import random
import einops
from torchvision.utils import save_image
import os


def flip_SAI(data, angRes):
    if len(data.shape) == 2:
        H, W = data.shape
        data = data.reshape(H, W, 1)

    H, W, C = data.shape
    data = data.reshape(angRes, H // angRes, angRes, W // angRes, C)  # [U, H, V, W, C]
    data = data[::-1, ::-1, ::-1, ::-1, :]
    data = data.reshape(H, W, C)

    return data


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = torch.flip(data, dims=[2])
        label = torch.flip(label, dims=[2])
    if random.random() < 0.5:  # flip along W-V direction
        data = torch.flip(data, dims=[1])
        label = torch.flip(label, dims=[1])
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.permute(0, 2, 1)
        label = label.permute(0, 2, 1)
    return data, label


def process_and_save_light_field_images(dataset_dir, patch_size, angRes_in, angRes_out, save_dir):
    # 获取文件夹中的所有 .h5 文件
    file_list = [f for f in os.listdir(dataset_dir) if f.endswith('.h5')]

    if not file_list:
        print("No .h5 files found in the specified directory.")
        return

    for index, file_name in enumerate(file_list):
        full_file_path = os.path.join(dataset_dir, file_name)
        with h5py.File(full_file_path, 'r') as hf:
            trans_LF = np.array(hf.get('trans_LF'))  # trans_LF
            blended_LF = np.array(hf.get('blended_LF'))  # 混合图像

            trans_LF = torch.from_numpy(trans_LF)
            blended_LF = torch.from_numpy(blended_LF)

            print(f"Original trans_LF shape: {trans_LF.shape}")
            print(f"Original blended_LF shape: {blended_LF.shape}")

            # Ensure dimensions are divisible by angRes_in
            C, H, W = trans_LF.shape
            new_H = (H // angRes_in) * angRes_in
            new_W = (W // angRes_in) * angRes_in
            trans_LF = trans_LF[:, :new_H, :new_W]
            blended_LF = blended_LF[:, :new_H, :new_W]

            print(f"Cropped trans_LF shape: {trans_LF.shape}")
            print(f"Cropped blended_LF shape: {blended_LF.shape}")

            window_size = patch_size

            '''crop'''
            trans_LF = einops.rearrange(trans_LF, 'c (u H) (v W) -> c (u v) H W', u=angRes_in, v=angRes_in)
            blended_LF = einops.rearrange(blended_LF, 'c (u H) (v W) -> c (u v) H W', u=angRes_in, v=angRes_in)

            _, _, H, W = trans_LF.size()
            x = random.randrange(0, H - window_size, 8)
            y = random.randrange(0, W - window_size, 8)

            trans_LF = trans_LF[:, :, x:x + window_size, y:y + window_size]  # [ah,aw,ph,pw]
            blended_LF = blended_LF[:, :, x:x + window_size, y:y + window_size]  # [ah,aw,ph,pw]

            trans_LF = einops.rearrange(trans_LF, 'c (u v) H W -> c (u H) (v W)', u=angRes_in, v=angRes_in)
            blended_LF = einops.rearrange(blended_LF, 'c (u v) H W -> c (u H) (v W)', u=angRes_in, v=angRes_in)

            '''augmentation'''
            trans_LF, blended_LF = augmentation(trans_LF, blended_LF)
            reflection_LF = blended_LF - trans_LF

        Lr_angRes_in = angRes_in
        Lr_angRes_out = angRes_out

        # Save images
        save_image(trans_LF, f"{save_dir}/trans_LF_{index}.png")
        save_image(blended_LF, f"{save_dir}/blended_LF_{index}.png")
        save_image(reflection_LF, f"{save_dir}/reflection_LF{index}.png")
    return blended_LF, trans_LF, [Lr_angRes_in, Lr_angRes_out]


# 示例用法
dataset_dir = r'D:\ml\LFDUNRR\LFRR_DATA\train\syn'
patch_size = 128
angRes_in = 5
angRes_out = 5
save_dir = r'D:\ml\LFDUNRR\processed_image'

blended_LF, trans_LF, [Lr_angRes_in, Lr_angRes_out] = process_and_save_light_field_images(dataset_dir, patch_size, angRes_in, angRes_out, save_dir)
print(blended_LF,trans_LF, [Lr_angRes_in, Lr_angRes_out])


