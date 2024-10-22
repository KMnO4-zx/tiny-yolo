from src.head import Detect

import torch

# 实例化并测试代码
def test_detect():
    ch = (32, 64, 128)  # 输入信道的数量，可以根据需要调整
    model = Detect(nc=80, ch=ch)  # 实例化检测模型
    model.eval()  # 设置为评估模式

    # 创建一些虚拟输入数据，模拟不同的检测层输出
    x = [torch.randn(1, c, 20, 20) for c in ch]  # 每层输入形状为 (batch_size, channels, height, width)

    # 前向传播，获取输出
    with torch.no_grad():
        output = model(x)
    print(output.shape)

test_detect()