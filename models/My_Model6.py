# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# Based on ConvNeXt, timm, DINO and DeiT code bases
# https://github.com/facebookresearch/ConvNeXt
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

import sys
import os
import torch.nn.functional as F

class CSA(nn.Module):
    def __init__(self,in_channel,reduction=16,kernel_size=7):
        super(CSA, self).__init__()
        #通道注意力机制
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        #空间注意力机制
       # self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=kernel_size ,stride=1,padding=kernel_size//2,bias=False)
        self.conv0 = nn.Conv2d(2, 2, 5, padding=2, groups=2)
        self.conv_spatial = nn.Conv2d(2, 2, 5, stride=1, padding=6, groups=2, dilation=3)
        self.conv1 = nn.Conv2d(2, 1, 1)
    def forward(self,x):
        #通道注意力机制
        sca=self.sca(x)
        sca=sca*x
        #空间注意力机制
        max_out,_=torch.max(sca,dim=1,keepdim=True)
        mean_out=torch.mean(sca,dim=1,keepdim=True)
        out=torch.cat((max_out,mean_out),dim=1)
        out=self.conv0(out)
        out = self.conv_spatial(out)
        out = self.conv1(out)
        #print(out.shape)
        out=out*sca
        return out

class large_kernel(nn.Module):
    def __init__(self,in_channels,lkernel_size=3,small_kernel=3,drop_out_rate=0.):
        super(large_kernel,self).__init__()
        out_channels=2*in_channels
        self.bn=nn.InstanceNorm2d(in_channels)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=1)
        self.lk=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=lkernel_size,
                        padding = lkernel_size // 2,stride=1, groups=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=small_kernel,
                                  stride=1, padding=small_kernel // 2, groups=out_channels, dilation=1)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, 1, 0, groups=1)
        self.conv4 = nn.Conv2d(out_channels, in_channels, 1, 1, 0, groups=1)
        # self.conv5 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
        #                           stride=1, padding=1, groups=out_channels, dilation=1)

        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )
        self.csa=CSA(in_channel=out_channels)
        self.nonlinear = nn.GELU()
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
       # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)

        #self.max=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
       # self.block4 = Block(out_channels*2, out_channels*2, 3, 1, start_with_relu=True, grow_first=True)

    def forward(self,x):

        inp=x
        out=self.bn(x)
        out=self.conv1(out)
        out=self.lk(out)

        out=self.conv2(out)
        out = self.nonlinear(out)
        out=self.csa(out)
        out = self.bn2(out)
        out=self.conv3(out)
        #out = self.conv5(out)
        out=self.nonlinear(out)
        out = self.conv4(out)
        out = self.dropout1(out)

        out = inp + out * self.beta

        return out

class SPNet_MD(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        large_kernel_sizes = [13, 13, 13, 3]

        chan = width
        for num in enc_blk_nums:
           # print(num)
            self.encoders.append(
                nn.Sequential(
                    *[large_kernel(chan,lkernel_size=large_kernel_sizes[enc_blk_nums.index(num)]) for _ in range(num)]
                )
            )
            # print(large_kernel_sizes[enc_blk_nums.index(num)])
            # print(num)
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[large_kernel(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[large_kernel(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp
        x = torch.clamp(x, min=-1, max=1)

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


