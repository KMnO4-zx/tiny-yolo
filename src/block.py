import torch
import torch.nn as nn
import torch.nn.functional as F


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """
    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        # 定义一个卷积层，输入通道为 c1，输出通道为1，卷积核大小为 1x1，并且不使用偏置项
        # requires_grad_(False) 表示不需要更新卷积层的权重（固定权重）
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        
        # 使用 torch.arange 创建一个从 0 到 c1-1 的序列，用于初始化卷积层的权重
        # 这些权重将通过 nn.Parameter 包装，并调整为适合卷积操作的形状 (1, c1, 1, 1)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        
        # 保存输入通道数 c1
        self.c1 = c1

    def forward(self, x):
        # 获取输入张量的形状信息：batch大小、通道数、锚点数量
        b, _, a = x.shape  # batch, channels, anchors
        
        # 将输入张量 x 重新调整形状为 (batch, 4, c1, anchors)
        # 然后将第2维（通道数维度）和第3维（类别维度）进行交换
        # 对交换后的结果在类别维度上（即第1维）应用softmax
        # 然后通过卷积层 self.conv 进行卷积操作，最后将结果 reshape 回原来的形状 (batch, 4, anchors)
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

if __name__ == "__main__":
    # 实例化 DFL 类，假设输入通道数为16
    dfl_layer = DFL(c1=16)

    # 创建一个随机的输入张量，形状为 (batch_size, c1 * 4, anchors)
    # 例如，batch_size 为 2，c1 为 16，anchors 为 32
    input_tensor = torch.randn(2, 16 * 4, 32)

    # 通过 DFL 层进行前向传播
    output = dfl_layer(input_tensor)

    # 输出张量的形状
    print(f"Output shape: {output.shape}")