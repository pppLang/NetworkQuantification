import torch
from torch import nn
from torch.autograd import Function as F
import time


class SignFunction(F):
    def forward(self, input):
        # print('forward')
        self.save_for_backward(input)
        return input.sign()

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = grad_output.clone()
            grad_input[input.ge(1)] = 0
            grad_input[input.le(-1)] = 0
            grad_input[input<0] = 2 + 2*grad_input[input<0]*2
            grad_input[input>=0] = 2 - 2*grad_input[input>=0]*2
        # print('backward')
        return grad_input

class ShortcutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ShortcutBlock, self).__init__()
        self.binconv = BinConv2d(in_channels, out_channels, kernel_size, 1, padding=int((kernel_size-1)/2))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        fea = SignFunction()(x)
        fea = self.binconv(fea)
        fea = self.bn(fea)
        return fea+x

class BinConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(BinConv2d, self).__init__()
        # self.sign = SignFunction()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
        padding=padding, dilation=dilation, groups=groups, bias=False)

        self.real_weight = self.conv.weight.data.clone().cuda()
        # self._parameters.append({})

    def forward(self, x):
        self.conv.weight.data = self.real_weight.sign()
        # fea = SignFunction()(x)
        fea = self.conv(x)
        return fea


def updataConvWei(model, lr):
    for layer in model.modules():
        if isinstance(layer, BinConv2d):

            # print(layer.conv.weight.shape)
            # print(layer.conv.weight)
            # print()
            # print(layer.conv.weight.grad.shape)
            # print(layer.conv.weight.grad)
            layer.real_weight.grad = torch.zeros_like(layer.real_weight)
            num = torch.sum(torch.ones_like(layer.real_weight))
            layer.real_weight.grad[torch.abs(layer.real_weight).lt(1)] = 1 * torch.mean(torch.abs(layer.real_weight))
            # print(layer.real_weight.grad)
            # print(layer.real_weight)
            layer.real_weight = layer.real_weight - lr*layer.real_weight.grad
            # print(layer.real_weight)
            # print()
            # time.sleep(10)