import torch
import torch.nn as nn
import copy
import math

from .block import DFL
from .conv import Conv
from .utils.tal import make_anchors


class Detect(nn.Module):
    """YOLOv8检测头，用于检测模型。"""

    dynamic = False  # 强制重新构建格子
    export = False  # 导出模式
    end2end = False  # 结束到结束模式
    max_det = 300  # 最多检测目标数量
    shape = None  # 用于记录输入形状
    anchors = torch.empty(0)  # 初始化锚点为空
    strides = torch.empty(0)  # 初始化身坐标常数为空

    def __init__(self, nc=80, ch=()):
        """
        初始化YOLOv8检测层，指定类别数量和信道数量。

        参数：
            nc (int): 类别数量，默认为80
            ch (tuple): 各检测层的信道数量
        """
        super().__init__()
        self.nc = nc  # 类别数量
        self.nl = len(ch)  # 检测层的数量
        self.reg_max = 16  # DFL信道，用于表示回归编码最大值
        self.no = nc + self.reg_max * 4  # 每个锚点的输出数量，类别数量+回归信息
        self.stride = torch.zeros(self.nl)  # 常数每个检测层的步长
        
        # 设置信道数量，按类别数和不同检测层输入的信息来确定
        c2 = max((16, ch[0] // 4, self.reg_max * 4))  # 用于目标框回归的信道
        c3 = max(ch[0], min(self.nc, 100))  # 用于目标类别识别的信道
        
        # 创建用于监控框的一个信道列表
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )  # 用于回归目标框的两层Conv和一个空间信道连接
        
        # 创建用于监控类别的一个信道列表
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )  # 用于识别目标类别的两层Conv和一个空间信道连接
        
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()  # 使用DFL的标识，如果reg_max大于1，则使用DFL

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)  # 深度复制这些模块，用于one2one的测量
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """
        进行目标框和类别概率的输出连接并返回。

        参数：
            x (list): 输入的不同级别的信息

        返回：
            返回一组用于表示目标框和类别的决定。
        """
        if self.end2end:
            return self.forward_end2end(x)  # 返回end2end模式的测量

        for i in range(self.nl):
            # 将回归和类别的输出合并为一个输出
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)  # 使用torch.cat将两个输出在信道维度上合并
        if self.training:  # 如果是训练模式
            return x  # 返回合并后的信息
        y = self._inference(x)  # 进行测试输出
        return y if self.export else (y, x)  # 返回并且根据export来确定是否返回输出

    def forward_end2end(self, x):
        """
        进行YOLOv8的end2end模式的前向通过。

        参数：
            x (tensor): 输入的信息。

        返回：
            (dict, tensor): 如果不是训练模式，返回一个包含one2many和one2one检测的输出。
                           如果是训练模式，返回一个包含one2many和one2one检测的输出。
        """
        # 将输入的信息做断连，使得后续的运算不会影响原来的输入数据
        x_detach = [xi.detach() for xi in x]  # 断开各种连接
        # 通过对one2one模块进行输出合并
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]  # 对象one到one的输出连接
        # 继续对常规的合并输出
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)  # 将信息连接并断
        if self.training:  # 训练路径
            return {"one2many": x, "one2one": one2one}  # 返回训练模式下的两组输出

        # 如果不是训练模式，则进行测试输出的后处理
        y = self._inference(one2one)  # 运行运行测量
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)  # 运行后编码进行处理
        return y if self.export else (y, {"one2many": x, "one2one": one2one})  # 返回结果

    def _inference(self, x):
        """
        连接并运行测试测试框和类别概率。
        """
        # 获得输入的形状信息，BCHW代表批次、信道、高度和宽度
        shape = x[0].shape  # BCHW 中应的内部信息
        # 将所有测试层的输出在信道维度合并，输出一个连接的tensor
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)  # 输出连接
        
        # 如果是动态或输入形状变化，则重新生成锚点和身坐标
        if self.dynamic or self.shape != shape:  # 确实格子
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))  # 磁的工具与动态模式
            self.shape = shape  # 更新当前形状

        # 如果是导出模式，且格式为saved_model等，则分解目标框和类别
        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # 避免构建FlexSplitV运行
            box = x_cat[:, : self.reg_max * 4]  # 目标框信息
            cls = x_cat[:, self.reg_max * 4 :]  # 类别概率
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)  # 将测试输出分解为目标框和类别信息

        # 如果是特定格式的导出，则进行前编码计算，为了提高数学稳定性
        if self.export and self.format in {"tflite", "edgetpu"}:  # 前存计算构因数存检
            grid_h = shape[2]  # 格子高度
            grid_w = shape[3]  # 格子宽度
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)  # 生成格子的大小
            norm = self.strides / (self.stride[0] * grid_size)  # 步长和格子尺寸的比值

