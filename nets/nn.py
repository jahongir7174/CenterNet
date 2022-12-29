import math

import torch


def fuse_conv(conv, norm):
    """
    [https://nenadmarkus.com/p/fusing-batchnorm-and-conv/]
    """
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.padding,
                                 conv.dilation,
                                 conv.groups, True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class ConvTranspose(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(in_ch, out_ch, k, s, p, 0, g, False, d)
        self.norm = torch.nn.BatchNorm2d(out_ch)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.add_m = s != 1 or in_ch != self.expansion * out_ch

        self.conv1 = Conv(in_ch, out_ch, torch.nn.ReLU(), 3, s, 1)
        self.conv2 = Conv(out_ch, out_ch, torch.nn.Identity(), 3, 1, 1)

        if self.add_m:
            self.conv3 = Conv(in_ch, out_ch, torch.nn.Identity(), s=s)

    def zero_init(self):
        torch.nn.init.zeros_(self.conv2.norm.weight)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_m:
            residual = self.conv3(x)

        return self.relu(out + residual)


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.add_m = s != 1 or in_ch != self.expansion * out_ch

        self.conv1 = Conv(in_ch, out_ch, torch.nn.ReLU())
        self.conv2 = Conv(out_ch, out_ch, torch.nn.ReLU(), 3, s, 1)
        self.conv3 = Conv(out_ch, out_ch * self.expansion, torch.nn.Identity())

        if self.add_m:
            self.conv4 = Conv(in_ch, self.expansion * out_ch, torch.nn.Identity(), s=s)

    def zero_init(self):
        torch.nn.init.zeros_(self.conv3.norm.weight)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.add_m:
            residual = self.conv4(x)

        return self.relu(out + residual)


class ResNet(torch.nn.Module):
    def __init__(self, block, depth):
        super().__init__()

        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []
        filters = [3, 64, 128, 256, 512]

        # p1/2
        self.p1.append(Conv(filters[0], filters[1], torch.nn.ReLU(), 7, 2, 3))
        # p2/4
        for i in range(depth[0]):
            if i == 0:
                self.p2.append(torch.nn.MaxPool2d(3, 2, 1))
                self.p2.append(block(filters[1], filters[1], 1))
            else:
                self.p2.append(block(block.expansion * filters[1], filters[1]))
        # p3/8
        for i in range(depth[1]):
            if i == 0:
                self.p3.append(block(block.expansion * filters[1], filters[2], 2))
            else:
                self.p3.append(block(block.expansion * filters[2], filters[2], 1))
        # p4/16
        for i in range(depth[2]):
            if i == 0:
                self.p4.append(block(block.expansion * filters[2], filters[3], 2))
            else:
                self.p4.append(block(block.expansion * filters[3], filters[3], 1))
        # p5/32
        for i in range(depth[3]):
            if i == 0:
                self.p5.append(block(block.expansion * filters[3], filters[4], 2))
            else:
                self.p5.append(block(block.expansion * filters[4], filters[4], 1))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

        for m in self.modules():
            if hasattr(m, 'zero_init'):
                m.zero_init()

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p1, p2, p3, p4, p5

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


class FPN(torch.nn.Module):
    def __init__(self, width):
        super().__init__()

        self.p4 = [Conv(width[3], width[2], torch.nn.ReLU(), 3, p=1),
                   ConvTranspose(width[2], width[2], torch.nn.ReLU(), 4, 2, 1)]
        self.p3 = [Conv(width[2], width[1], torch.nn.ReLU(), 3, p=1),
                   ConvTranspose(width[1], width[1], torch.nn.ReLU(), 4, 2, 1)]
        self.p2 = [Conv(width[1], width[0], torch.nn.ReLU(), 3, p=1),
                   ConvTranspose(width[0], width[0], torch.nn.ReLU(), 4, 2, 1)]

        self.p4 = torch.nn.Sequential(*self.p4)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p2 = torch.nn.Sequential(*self.p2)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()
            if isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, torch.nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]

    def forward(self, x):
        p1, p2, p3, p4, p5 = x
        p4 = self.p4(p5)
        p3 = self.p3(p4)
        p2 = self.p2(p3)
        return p2


class Head(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.wh = torch.nn.Conv2d(in_channels, 2, 1)
        self.offset = torch.nn.Conv2d(in_channels, 2, 1)
        self.center = torch.nn.Conv2d(in_channels, num_classes, 1)

        bias_init = float(-math.log((1 - 0.1) / 0.1))
        self.center.bias.data.fill_(bias_init)
        for head in [self.wh, self.offset]:
            for m in head.modules():
                if isinstance(m, torch.nn.Conv2d):
                    if hasattr(m, 'weight') and m.weight is not None:
                        torch.nn.init.normal_(m.weight, 0, 0.001)
                    if hasattr(m, 'bias') and m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        wh = self.wh(x)
        offset = self.offset(x)
        center = self.center(x)
        return center.sigmoid(), wh, offset


class CenterNet(torch.nn.Module):
    def __init__(self, block, depth, num_classes):
        super().__init__()
        filters = [64, 128, 256, 512] * block.expansion
        self.net = ResNet(block, depth)
        self.fpn = FPN(filters)

        self.head = Head(filters[0], num_classes)

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(x)


def center_net_18(num_classes: int = 80):
    return CenterNet(BasicBlock, [2, 2, 2, 2], num_classes)


def center_net_34(num_classes: int = 80):
    return CenterNet(BasicBlock, [3, 4, 6, 3], num_classes)


def center_net_50(num_classes: int = 80):
    return CenterNet(Bottleneck, [3, 4, 6, 3], num_classes)


def center_net_101(num_classes: int = 80):
    return CenterNet(Bottleneck, [3, 4, 23, 3], num_classes)


def center_net_152(num_classes: int = 80):
    return CenterNet(Bottleneck, [3, 8, 36, 3], num_classes)


def center_net_200(num_classes: int = 80):
    return CenterNet(Bottleneck, [3, 24, 36, 3], num_classes)
