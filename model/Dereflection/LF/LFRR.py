import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from .swin import *
import einops
from .warpnet import Net
from .disparity_utils import warp_to_ref_view_parallel, warp, back_projection_from_HR_ref_view
from .LFRRN_utils import *
from .vgg import Vgg19
from model import common
from .uti import *
import pytorch_ssim
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.functional import grid_sample
import random


def update_S(A, cen_fea, CG, NL, Disparity):
    A_out = A.clone()
    global_fea = einops.rearrange(A_out, 'b an2 c h w -> b c an2 h w')
    warped = feature_warp_to_ref_view_parallel(global_fea, Disparity, refPos=[1, 1])
    warped = einops.rearrange(warped, 'b c an2 h w -> b an2 c h w')
    centra_view = NL(warped, cen_fea)
    warped_centra_fea = back_projection_from_HR_ref_view(centra_view, refPos=[1, 1], disparity=Disparity,
                                                         angular_resolution=3, scale=1)
    ST1 = CG(warped_centra_fea, A)
    ST = ST1.clone()
    ST[:, 4] = centra_view
    return ST


class blockNL(nn.Module):
    def __init__(self, channels=48, angRes=5, fs=9, n=9):
        super(blockNL, self).__init__()
        self.channels = channels
        self.angRes = angRes
        self.fs = fs
        self.n = n
        self.softmax = nn.Softmax(dim=-1)

        # 注意力输入变换，使用 GroupNorm 替代 BatchNorm
        self.t = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        # 融合多个视图结果
        self.W = nn.Conv2d(channels * n, channels, kernel_size=1, bias=False)



    def forward(self, x, center):
        B, n, C, H, W = x.shape
        Sx_list = []
        center_view = center  # [B, C, H, W]

        theta = self.t(center_view).permute(0, 2, 3, 1).unsqueeze(-2)  # [B, H, W, 1, C]

        for i in range(n):
            x_fea = x[:, i]  # [B, C, H, W]

            phi = self.p(x_fea)  # [B, C, H, W]
            b, c, h, w = phi.size()
            phi_patches = F.unfold(phi, self.fs, padding=self.fs // 2)  # [B, C*fs*fs, HW]
            phi_patches = phi_patches.view(b, c, self.fs * self.fs, h, w)  # [B, C, fs*fs, H, W]
            phi_patches = phi_patches.permute(0, 3, 4, 1, 2)  # [B, H, W, C, fs*fs]

            att = torch.matmul(theta, phi_patches)  # [B, H, W, 1, fs*fs]
            att = self.softmax(att)  # attention

            g = self.g(x_fea)  # [B, C, H, W]
            g_patches = F.unfold(g, self.fs, padding=self.fs // 2)
            g_patches = g_patches.view(b, c, self.fs * self.fs, h, w)
            g_patches = g_patches.permute(0, 3, 4, 2, 1)  # [B, H, W, fs*fs, C]

            out_x = torch.matmul(att, g_patches).squeeze(-2)  # [B, H, W, C]
            out_x = out_x.permute(0, 3, 1, 2)  # [B, C, H, W]

            Sx = self.w(out_x) + center_view  # 残差连接
            Sx_list.append(Sx)

        Sx_stack = torch.stack(Sx_list, dim=1)  # [B, n, C, H, W]
        Sx_stack = Sx_stack.permute(0, 2, 1, 3, 4).reshape(B, C * n, H, W)
        enhanced = self.W(Sx_stack)  # 最终融合

        return enhanced


class blockNL1(nn.Module):
    def __init__(self, channels=48, angRes=5, fs=9, n=9):
        super(blockNL1, self).__init__()
        self.channels = channels
        self.angRes = angRes
        self.fs = fs
        self.n = n
        self.softmax = nn.Softmax(dim=-1)

        self.t = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        # 融合多个视图结果
        self.W = nn.Conv2d(channels * n, channels, kernel_size=1, bias=False)

    def forward(self, x, center):
        B, n, C, H, W = x.shape
        Sx_list = []
        center_view = center  # [B, C, H, W]

        theta = self.t(center_view).permute(0, 2, 3, 1).unsqueeze(-2)  # [B, H, W, 1, C]

        for i in range(n):
            x_fea = x[:, i]  # [B, C, H, W]

            phi = self.p(x_fea)  # [B, C, H, W]
            b, c, h, w = phi.size()
            phi_patches = F.unfold(phi, self.fs, padding=self.fs // 2)  # [B, C*fs*fs, HW]
            phi_patches = phi_patches.view(b, c, self.fs * self.fs, h, w)  # [B, C, fs*fs, H, W]
            phi_patches = phi_patches.permute(0, 3, 4, 1, 2)  # [B, H, W, C, fs*fs]

            att = torch.matmul(theta, phi_patches)  # [B, H, W, 1, fs*fs]
            att = self.softmax(att)  # attention

            g = self.g(x_fea)  # [B, C, H, W]
            g_patches = F.unfold(g, self.fs, padding=self.fs // 2)
            g_patches = g_patches.view(b, c, self.fs * self.fs, h, w)
            g_patches = g_patches.permute(0, 3, 4, 2, 1)  # [B, H, W, fs*fs, C]

            out_x = torch.matmul(att, g_patches).squeeze(-2)  # [B, H, W, C]
            out_x = out_x.permute(0, 3, 1, 2)  # [B, C, H, W]

            Sx = self.w(out_x) + center_view  # 残差连接
            Sx_list.append(Sx)

        Sx_stack = torch.stack(Sx_list, dim=1)  # [B, n, C, H, W]
        Sx_stack = Sx_stack.permute(0, 2, 1, 3, 4).reshape(B, C * n, H, W)
        enhanced = self.W(Sx_stack)  # 最终融合

        return enhanced


class CTG(nn.Module):
    def __init__(self, channels=48, angRes=5, fs=9):
        super(CTG, self).__init__()
        self.channels = channels
        self.angRes = angRes
        self.fs = fs
        self.softmax = nn.Softmax(dim=-1)

        self.t = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)

    def forward(self, warped_c, A):
        B, n, C, H, W = A.shape
        A_out = A.clone()

        for i in range(n):
            x_fea = A[:, i]  # 当前视图
            x_fea1 = warped_c[:, i]  # 中心视图 warp 后的特征（对应当前视图）

            # 注意力 query
            theta = self.t(x_fea).permute(0, 2, 3, 1).unsqueeze(-2)  # [B, H, W, 1, C]

            # 注意力 key
            phi = self.p(x_fea1)
            b, c, h, w = phi.size()
            phi_patches = F.unfold(phi, self.fs, padding=self.fs // 2)  # [B, C*fs*fs, HW]
            phi_patches = phi_patches.view(b, c, self.fs * self.fs, h, w)  # [B, C, fs*fs, H, W]
            phi_patches = phi_patches.permute(0, 3, 4, 1, 2)  # [B, H, W, C, fs*fs]

            att = torch.matmul(theta, phi_patches)  # [B, H, W, 1, fs*fs]
            att = self.softmax(att)

            # 注意力 value
            g = self.g(x_fea1)
            g_patches = F.unfold(g, self.fs, padding=self.fs // 2)  # [B, C*fs*fs, HW]
            g_patches = g_patches.view(b, self.channels, self.fs * self.fs, h, w)  # [B, C, fs*fs, H, W]
            g_patches = g_patches.permute(0, 3, 4, 2, 1)  # [B, H, W, fs*fs, C]

            out_x = torch.matmul(att, g_patches).squeeze(-2)  # [B, H, W, C]
            out_x = out_x.permute(0, 3, 1, 2)  # [B, C, H, W]
            Sx = self.w(out_x) + x_fea  # 残差连接
            A_out[:, i] = Sx

        return A_out


class CTG1(nn.Module):
    def __init__(self, channels=48, angRes=5, fs=9):
        super(CTG1, self).__init__()
        self.channels = channels
        self.angRes = angRes
        self.fs = fs
        self.softmax = nn.Softmax(dim=-1)

        self.t = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)

    def forward(self, warped_c, A):
        B, n, C, H, W = A.shape
        A_out = A.clone()

        for i in range(n):
            x_fea = A[:, i]  # 当前视图
            x_fea1 = warped_c[:, i]  # 中心视图 warp 后的特征（对应当前视图）

            # 注意力 query
            theta = self.t(x_fea).permute(0, 2, 3, 1).unsqueeze(-2)  # [B, H, W, 1, C]

            # 注意力 key
            phi = self.p(x_fea1)
            b, c, h, w = phi.size()
            phi_patches = F.unfold(phi, self.fs, padding=self.fs // 2)  # [B, C*fs*fs, HW]
            phi_patches = phi_patches.view(b, c, self.fs * self.fs, h, w)  # [B, C, fs*fs, H, W]
            phi_patches = phi_patches.permute(0, 3, 4, 1, 2)  # [B, H, W, C, fs*fs]

            att = torch.matmul(theta, phi_patches)  # [B, H, W, 1, fs*fs]
            att = self.softmax(att)

            # 注意力 value
            g = self.g(x_fea1)
            g_patches = F.unfold(g, self.fs, padding=self.fs // 2)  # [B, C*fs*fs, HW]
            g_patches = g_patches.view(b, self.channels, self.fs * self.fs, h, w)  # [B, C, fs*fs, H, W]
            g_patches = g_patches.permute(0, 3, 4, 2, 1)  # [B, H, W, fs*fs, C]

            out_x = torch.matmul(att, g_patches).squeeze(-2)  # [B, H, W, C]
            out_x = out_x.permute(0, 3, 1, 2)  # [B, C, H, W]
            Sx = self.w(out_x) + x_fea  # 残差连接
            A_out[:, i] = Sx

        return A_out


class DeepUnfoldingNet(nn.Module):
    def __init__(self, args):
        super(DeepUnfoldingNet, self).__init__()
        self.angRes = 3
        self.num_iter = args.iterations
        self.channel = args.channel
        self.layers = args.layers
        self.batch = args.batch_size
        channel = args.channel
        self.device = torch.device(args.device)
        self.fs = args.fs
        dispchannel = 32
        t = self.num_iter
        self.n = 9

        # 视差
        self.DispFeaExtract = FeaExtract(dispchannel)
        self.disp_net = Net(self.angRes)

        # nonlocal
        self.NL = blockNL(channels=3, angRes=self.angRes, fs=self.fs, n=self.n)
        self.NL1 = blockNL1(channels=3, angRes=self.angRes, fs=self.fs, n=self.n)
        self.CG = CTG(channels=3, angRes=self.angRes, fs=self.fs)
        self.CG1 = CTG1(channels=3, angRes=self.angRes, fs=self.fs)

        # Unet
        self.Encoding_block1 = Encoding_Block(64)
        self.Encoding_block2 = Encoding_Block(64)
        self.Encoding_block3 = Encoding_Block(64)
        self.Encoding_block4 = Encoding_Block(64)

        self.Encoding_block_end = Encoding_Block_End(64)

        self.Decoding_block1 = Decoding_Block(256)
        self.Decoding_block2 = Decoding_Block(256)
        self.Decoding_block3 = Decoding_Block(256)
        self.Decoding_block4 = Decoding_Block(256)

        self.feature_decoding_end = Decoding_Block_End(256)

        self.act = nn.ReLU()

        self.construction = nn.Conv2d(64, 3, 3, padding=1)

        G0 = 64
        kSize = 3

        self.Fe_e = nn.ModuleList([nn.Sequential(*[
            nn.Conv2d(3, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ]) for _ in range(t)])

        self.RNNF = nn.ModuleList([nn.Sequential(*[
            nn.Conv2d((i + 2) * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            self.act,
            nn.Conv2d(64, 3, 3, padding=1)

        ]) for i in range(t)])

        self.Fe_f = nn.ModuleList(
            [nn.Sequential(*[nn.Conv2d((2 * i + 3) * G0, G0, 1, padding=0, stride=1)]) for i in range(t - 1)])

        # 步长参数
        self.alpha1 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=self.device, requires_grad=True))
        self.alpha2 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=self.device, requires_grad=True))
        self.alpha3 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=self.device, requires_grad=True))
        self.alpha4 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=self.device, requires_grad=True))
        self.u = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=self.device, requires_grad=True))
        self.y1 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=self.device, requires_grad=True))
        self.y2 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=self.device, requires_grad=True))
        self.n1 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=self.device, requires_grad=True))
        self.n2 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=self.device, requires_grad=True))

        # 初始化透射率
        self.alpha = nn.Parameter(torch.tensor(0.8))  # 或随机设定 (0.6 ~ 0.9)
        self.w1 = nn.Parameter(torch.tensor(0.5))  # 或随机设定 (0.6 ~ 0.9)

    def forward(self, data, data_info):
        # print(data.shape)


        b, _, H, W = data.shape
        x_mv = self.LFsplit(data, self.angRes)

        disp_fea_initial = self.DispFeaExtract(x_mv)
        disp_fea_initial = einops.rearrange(disp_fea_initial, 'b (a1 a2) c h w -> b c (a1 h) (a2 w)', a1=self.angRes,
                                            a2=self.angRes)
        Disparity = self.disp_net(disp_fea_initial).expand(-1, 2, -1, -1)
        I = x_mv

        T = self.w1 * I
        R = (I - T) / self.alpha
        centra_fea_I = x_mv[:, 4]
        centra_fea_T = self.w1 * centra_fea_I
        centra_fea_R = (centra_fea_I - centra_fea_T) / self.alpha
        fea_list = []
        V_list = []
        fea_list1 = []
        V_list1 = []

        for k in range(self.num_iter):
            x = einops.rearrange(T, 'B (u v) c h w -> B  c (u h) (v w)', u=3, v=3)
            fea = self.Fe_e[k](x)
            fea_list.append(fea)
            if k != 0:
                fea = self.Fe_f[k - 1](torch.cat(fea_list, 1))

            encode0, down0 = self.Encoding_block1(fea)
            encode1, down1 = self.Encoding_block2(down0)
            encode2, down2 = self.Encoding_block3(down1)
            encode3, down3 = self.Encoding_block4(down2)

            media_end = self.Encoding_block_end(down3)

            decode3 = self.Decoding_block1(media_end, encode3)
            decode2 = self.Decoding_block2(decode3, encode2)
            decode1 = self.Decoding_block3(decode2, encode1)
            decode0 = self.feature_decoding_end(decode1, encode0)
            fea_list.append(decode0)
            V_list.append(decode0)
            if k == 0:
                decode0 = self.construction(self.act(decode0))
            else:
                decode0 = self.RNNF[k - 1](torch.cat(V_list, 1))

            vT = x + decode0
            vT = einops.rearrange(vT, 'B  c (u h) (v w) -> B (u v) c h w', u=3, v=3)


            x1 = einops.rearrange(R, 'B (u v) c h w -> B  c (u h) (v w)', u=3, v=3)
            fea1 = self.Fe_e[k](x1)
            fea_list1.append(fea1)
            if k != 0:
                fea1 = self.Fe_f[k - 1](torch.cat(fea_list1, 1))
            encode0, down0 = self.Encoding_block1(fea1)
            encode1, down1 = self.Encoding_block2(down0)
            encode2, down2 = self.Encoding_block3(down1)
            encode3, down3 = self.Encoding_block4(down2)

            media_end = self.Encoding_block_end(down3)

            decode3 = self.Decoding_block1(media_end, encode3)
            decode2 = self.Decoding_block2(decode3, encode2)
            decode1 = self.Decoding_block3(decode2, encode1)
            decode0 = self.feature_decoding_end(decode1, encode0)
            fea_list1.append(decode0)
            V_list1.append(decode0)
            if k == 0:
                decode0 = self.construction(self.act(decode0))
            else:
                decode0 = self.RNNF[k - 1](torch.cat(V_list1, 1))
            vR = x1 + decode0
            vR = einops.rearrange(vR, 'B  c (u h) (v w) -> B (u v) c h w', u=3, v=3)

            ST = update_S(vT, centra_fea_T, self.CG, self.NL, Disparity)
            SR = update_S(vR, centra_fea_R, self.CG1, self.NL1, Disparity)
            eR = SR - vR
            eT = ST - vT

            eR = eR - self.alpha1 * (self.u * (I - T - eT - self.alpha * (R + eR)) + self.y2 * (SR - R - eR))
            eT = eT - self.alpha2 * (self.u * (I - T - eT - self.alpha * (R - eR)) + self.y1 * (ST - T - eT))
            R = R - self.alpha4 * (
                    I - T - self.alpha * R + self.u * (I - T - eT - self.alpha * (R + eR)) + self.y2 * (
                    SR - R - eR) + self.n2 * (vR - R))
            T = T - self.alpha3 * (
                    I - T - self.alpha * R + self.u * (I - T - eT - self.alpha * (R + eR)) + self.y1 * (
                    ST - T - eT) + self.n1 * (vT - T))

            centra_fea_T = T[:, 4]
            centra_fea_R = R[:, 4]

        out = einops.rearrange(T, 'B( a1 a2) c h w -> B c (a1 h) (a2 w)', a1=3, a2=3)
        return Disparity, out

    @staticmethod
    def LFsplit(data, angRes):
        """用于将一个光场（Light Field, LF）数据分割成多个视角的子图像。"""
        b, _, H, W = data.shape
        h = int(H / angRes)
        w = int(W / angRes)
        data_sv = []
        for u in range(angRes):
            for v in range(angRes):
                data_sv.append(data[:, :, u * h:(u + 1) * h, v * w:(v + 1) * w])
        data_st = torch.stack(data_sv, dim=1)
        '''data_sv 将包含 25 个子图像，每个子图像的形状为 [1, 3, 128, 128]  最终返回的 data_st 的形状为 [1, 25, 3, 128, 128]'''
        return data_st



class Encoding_Block(nn.Module):
    def __init__(self, c_in):
        super(Encoding_Block, self).__init__()

        self.down = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=3 // 2)

        self.act = nn.ReLU()
        body = [nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2),nn.ReLU(),
                common.ResBlock(common.default_conv, 64, 3),common.ResBlock(common.default_conv, 64, 3),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=3 // 2)]
        self.body = nn.Sequential(*body)

    def forward(self, input):

        f_e = self.body(input)
        down = self.act(self.down(f_e))
        return f_e, down


class Encoding_Block_End(nn.Module):
    def __init__(self, c_in):
        super(Encoding_Block_End, self).__init__()

        self.down = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=3 // 2)
        self.act = nn.ReLU()
        head = [nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU()]
        body = [
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),

                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),

                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),
                ]
        tail = [nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=3 // 2)]
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
    def forward(self, input):
        out = self.head(input)
        f_e = self.body(out) + out
        f_e = self.tail(f_e)
        return f_e


class Decoding_Block(nn.Module):
    def __init__(self, c_in ):
        super(Decoding_Block, self).__init__()
        #self.up = torch.nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.up = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()
        body = [nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=1 // 2) ]
        self.body = nn.Sequential(*body)


    def forward(self, input, map):

        up = self.act(self.up(input,output_size=[input.shape[0], input.shape[1], map.shape[2], map.shape[3]]))
        out = torch.cat((up, map), 1)
        out = self.body(out)

        return out


class Decoding_Block_End(nn.Module):
    def __init__(self, c_in):
        super(Decoding_Block_End, self).__init__()
        # self.up = torch.nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.up = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()
        body = [nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),

                ]
        self.body = nn.Sequential(*body)



    def forward(self, input, map):
        up = self.act(self.up(input, output_size=[input.shape[0], input.shape[1], map.shape[2], map.shape[3]]))
        out = torch.cat((up, map), 1)
        out = self.body(out)
        return out


class FeaExtract(nn.Module):
    def __init__(self, channel):
        super(FeaExtract, self).__init__()
        self.FEconv = nn.Conv2d(3, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.FERB_1 = ResASPP(channel)
        self.FERB_2 = RB(channel)
        self.FERB_3 = ResASPP(channel)
        self.FERB_4 = RB(channel)

    def forward(self, x_mv):
        b, n, r, h, w = x_mv.shape
        x_mv = x_mv.contiguous().view(b * n, -1, h, w)
        intra_fea_0 = self.FEconv(x_mv)
        intra_fea = self.FERB_1(intra_fea_0)
        intra_fea = self.FERB_2(intra_fea)
        intra_fea = self.FERB_3(intra_fea)
        intra_fea = self.FERB_4(intra_fea)
        _, c, h, w = intra_fea.shape
        intra_fea = intra_fea.unsqueeze(1).contiguous().view(b, -1, c, h,
                                                             w)  # .permute(0,2,1,3,4)  # intra_fea:  B, N, C, H, W

        return intra_fea


class Extract_inter_fea(nn.Module):
    def __init__(self, channel, angRes):
        super(Extract_inter_fea, self).__init__()
        # 定义初始卷积层，将多视角特征合并
        self.FEconv = nn.Sequential(
            nn.Conv2d(angRes * angRes * 3, channel * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0, bias=False))
        # 定义残差注意力空洞空间金字塔池化层
        self.FERB_1 = ResASPP(channel)
        # 定义残差块
        self.FERB_2 = RB(channel)
        # 再次定义残差注意力空洞空间金字塔池化层
        self.FERB_3 = ResASPP(channel)
        # 再次定义残差块
        self.FERB_4 = RB(channel)

    def forward(self, x_mv):
        b, n, r, h, w = x_mv.shape
        x_mv = x_mv.contiguous().view(b, -1, h, w)
        # 初始特征提取
        inter_fea_0 = self.FEconv(x_mv)
        # 应用第一个残差注意力空洞空间金字塔池化层
        inter_fea = self.FERB_1(inter_fea_0)
        # 应用第一个残差块
        inter_fea = self.FERB_2(inter_fea)
        # 应用第二个残差注意力空洞空间金字塔池化层
        inter_fea = self.FERB_3(inter_fea)
        # 应用第二个残差块
        inter_fea = self.FERB_4(inter_fea)
        return inter_fea


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class disp_loss(nn.Module):
    def __init__(self):
        super(disp_loss, self).__init__()
        self.L1_Loss = torch.nn.L1Loss()
        self.L2_Loss = torch.nn.MSELoss()
        self.TVLoss = TVLoss()

    def loss_disp_smoothness(self, disp, img):
        img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
        weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
        weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

        loss = (((disp[:, :, :, :-1] - disp[:, :, :, 1:]).abs() * weight_x).sum() +
                ((disp[:, :, :-1, :] - disp[:, :, 1:, :]).abs() * weight_y).sum()) / \
               (weight_x.sum() + weight_y.sum())
        return loss

    def forward(self, disparity, gt, angular_res):
        refPos = [angular_res // 2, angular_res // 2]
        gt = einops.rearrange(gt, 'b c (u h) (v w) -> b c (u v) h w', u=angular_res, v=angular_res)

        PSV = feature_warp_to_ref_view_parallel(gt, disparity, refPos)
        cnter_view = gt[:, :, refPos[0] * angular_res + refPos[1]]
        loss = 0.
        for view in range(angular_res * angular_res):
            loss += self.L1_Loss(PSV[:, :, view], cnter_view) + self.L2_Loss(PSV[:, :, view], cnter_view) * 0.1

        loss1 = self.TVLoss(disparity) * 0.005
        loss2 = self.loss_disp_smoothness(disparity, cnter_view) * 0.1

        return loss + loss1 + loss2


class grad_loss(nn.Module):
    def __init__(self, args):
        super(grad_loss, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        self.kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        self.kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.angRes = args.angRes_in
        self.criterion_Loss = torch.nn.L1Loss()
        # self.criterion_Loss = torch.nn.MSELoss()

    def forward(self, SRt, HRt):
        self.weight_h = nn.Parameter(data=self.kernel_h, requires_grad=False).to(SRt.device)
        self.weight_v = nn.Parameter(data=self.kernel_v, requires_grad=False).to(SRt.device)

        # yv
        l0 = 0.
        for i in range(3):
            SR = SRt[:, i:i + 1]
            HR = HRt[:, i:i + 1]
            SR = einops.rearrange(SR, 'b c (a1 h) (a2 w) -> (b a1 a2) c h w', a1=3, a2=3)
            HR = einops.rearrange(HR, 'b c (a1 h) (a2 w) -> (b a1 a2) c h w', a1=3, a2=3)
            SR_v = F.conv2d(SR, self.weight_v, padding=2)
            HR_v = F.conv2d(HR, self.weight_v, padding=2)
            l1 = self.criterion_Loss(SR_v, HR_v)
            SR_h = F.conv2d(SR, self.weight_h, padding=2)
            HR_h = F.conv2d(HR, self.weight_h, padding=2)
            l2 = self.criterion_Loss(SR_h, HR_h)
            l0 = l0 + l1 + l2

        return l0


class VGGLoss(nn.Module):
    def __init__(self, device='cuda', vgg=None, weights=None, indices=None, normalize=False):
        super(VGGLoss, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8]
        self.indices = indices or [2, 7]
        self.device = device
        if normalize:
            self.normalize = None
        else:
            self.normalize = None
        print("Vgg: Weights: ", self.weights, " indices: ", self.indices, " normalize: ", self.normalize)

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss


class SSIMLoss(pytorch_ssim.SSIM):
    def forward(self, SR, HR):
        SR = einops.rearrange(SR, 'b c (a1 h) (a2 w) -> (b a1 a2) c h w', a1=3, a2=3)
        HR = einops.rearrange(HR, 'b c (a1 h) (a2 w) -> (b a1 a2) c h w', a1=3, a2=3)
        return 1. - super().forward(SR, HR)


# class SSIMLoss(pytorch_ssim.SSIM):
#     def forward(self, SR, HR):
#         SR = einops.rearrange(SR, 'b c (a1 h) (a2 w) -> (b a1 a2) c h w', a1=1, a2=6)
#         HR = einops.rearrange(HR, 'b c (a1 h) (a2 w) -> (b a1 a2) c h w', a1=1, a2=6)
#         return 1. - super().forward(SR, HR)


class saptial_loss(nn.Module):
    def __init__(self, args):
        super(saptial_loss, self).__init__()
        # self.L1_Loss = torch.nn.L1Loss()
        self.L2_Loss = torch.nn.MSELoss()
        self.grad_loss = grad_loss(args)
        self.vgg_loss = VGGLoss()
        self.SSIMLoss = SSIMLoss()
        self.TVLoss = TVLoss(TVLoss_weight=1.0)  # 添加 TVLoss

    def forward(self, result, gt, info=None):
        loss1 = self.L2_Loss(result, gt)
        loss2 = self.SSIMLoss(result, gt)
        loss3 = self.vgg_loss(result, gt)
        loss4 = self.grad_loss(result, gt)
        # loss5 = self.TVLoss(result)  # TVLoss
        loss = loss1 + loss2 + 0.1 * loss3 + 0.5 * loss4
        # loss = loss1 + loss2 + 0.1 * loss3 + 0.5 * loss4 + 0.02 * loss5
        return loss


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.saptial_loss = saptial_loss(args)
        self.disp_loss = disp_loss()
        # self.angRes = args.angRes_in
        self.angRes = 3

    def forward(self, disparity, result, gt, info=None):
        loss1 = self.saptial_loss(result, gt)
        loss2 = self.disp_loss(disparity, gt, self.angRes) * 0.1
        loss = loss1 + loss2
        return loss


def weights_init(m):
    pass
