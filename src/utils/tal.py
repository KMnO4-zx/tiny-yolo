import torch
import torch.nn as nn

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    # 用于生成锚点的函数

    anchor_points, stride_tensor = [], []  # 初始化用于存储锚点坐标和步幅张量的列表
    assert feats is not None  # 确保输入特征图不为空

    # 获取特征图的设备和数据类型，假设所有特征图具有相同的 dtype 和 device
    dtype, device = feats[0].dtype, feats[0].device

    # 遍历每个特征图及其对应的步幅
    for i, stride in enumerate(strides):
        # 获取当前特征图的形状 (batch_size, channels, height, width)
        _, _, h, w = feats[i].shape
        
        # 生成x方向和y方向上的偏移量
        # torch.arange生成从0到w-1的序列，再加上grid_cell_offset得到实际坐标
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        
        # 创建网格，使sx和sy对应每个网格点的x和y坐标
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        
        # 将网格坐标堆叠到一起，形成每个网格点的 (x, y) 坐标，并展平为形状 (h*w, 2)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        
        # 创建一个步幅张量，与当前特征图大小一致，每个位置都填充对应的步幅值
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    # 将所有特征图的锚点和步幅张量拼接在一起
    return torch.cat(anchor_points), torch.cat(stride_tensor)