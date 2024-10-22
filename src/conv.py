import math

import numpy as np
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    # 当d > 1时，计算扩张卷积的实际核大小（kernel size）
    # 如果k是一个整数，使用公式 d * (k - 1) + 1 计算实际核大小
    # 如果k是一个列表，逐元素应用公式 d * (x - 1) + 1 计算实际的核大小
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # 计算扩张卷积的实际核大小
    
    # 如果p为None，自动设置padding以确保输出的尺寸与输入相同（即"same" padding）
    # 如果k是一个整数，padding取k的一半（//表示整除）
    # 如果k是一个列表，逐元素计算padding，取每个元素的一半
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 自动填充padding
    
    # 返回计算的padding值
    return p

class Conv(nn.Module):
    # 默认激活函数为SiLU激活函数
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        # 调用父类nn.Module的初始化函数
        super().__init__()
        # 定义2D卷积层，使用autopad自动计算padding
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # 定义批归一化层，输入通道数为c2
        self.bn = nn.BatchNorm2d(c2)
        # 定义激活函数，若act为True，则使用默认激活函数SiLU；
        # 若act是nn.Module的实例，则使用传入的激活函数；
        # 否则，使用nn.Identity()，即不做激活
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # 按顺序应用卷积、批归一化和激活函数
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        # 用于推理时不需要批归一化，直接应用卷积和激活函数
        return self.act(self.conv(x))
    


if __name__ == '__main__':
    # 实例化 Conv 类
    conv_layer = Conv(c1=3, c2=1, k=1, s=1, act=True)  # 输入通道数为3，输出通道数为16，卷积核大小为3x3

    # 创建一个随机的输入张量，形状为 (batch_size, c1, height, width)
    # 假设 batch_size=1, 输入通道为3, 高度和宽度都为32
    input_tensor = torch.randn(1, 3, 32, 32)

    # 通过卷积层进行前向传播
    output = conv_layer(input_tensor)

    # 输出张量的形状
    print(f"Output shape: {output.shape}")