import torch
import torch.nn as nn



class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, activation=nn.Sequential()):
        padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            activation
        )

class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SELayer, self).__init__()
        self.SE = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(in_channels // reduction),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                )
    def forward(self, x):
        atten = self.SE(x)
        atten = torch.clamp(atten + 3, 0, 6) / 6
        return x * atten

class HardSwish(nn.Module):
	def __init__(self):
		super(HardSwish, self).__init__()

	def forward(self, inputs):
		clip = torch.clamp(inputs + 3, 0, 6) / 6
		return inputs * clip

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, ksize, stride, useSE, activation):
        super(InvertedResidual, self).__init__()
        self.is_shortcut = (stride == 1 and in_channels == out_channels)
        conv = [
            ConvBN(in_channels, mid_channels, 1, activation=activation),
            ConvBN(mid_channels, mid_channels, 3, stride, groups=mid_channels, activation=activation),
            SELayer(mid_channels) if useSE else nn.Sequential(),
            ConvBN(mid_channels, out_channels, 1)
        ]
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        out = self.conv(x)
        if self.is_shortcut:
            out = out + x
        return out

class MobileNetV3(nn.Module):
    def __init__(self, model_size='Large', width_mult=1.0, n_class=1000):
        """
        MobileNet V2

        Args:
            model_size (string): Model size - Large and Small for MobileNet V3
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            num_classes (int): Number of classes

        """
        super(MobileNetV3, self).__init__()

        RE = nn.ReLU(inplace=True)
        HS = HardSwish()
        in_channels = 16
        if model_size == 'Large':
            last_channel = 960, 1280
            inverted_residual_setting = [
                #k, out, exp, s, se, nl
                [3, 16,  16,  1, False, RE],
                [3, 24,  64,  2, False, RE],
                [3, 24,  72,  1, False, RE],
                [5, 40,  72,  2, True,  RE],
                [5, 40,  120, 1, True,  RE],
                [5, 40,  120, 1, True,  RE],
                [3, 80,  240, 2, False, HS],
                [3, 80,  200, 1, False, HS],
                [3, 80,  184, 1, False, HS],
                [3, 80,  184, 1, False, HS],
                [3, 112, 480, 1, True,  HS],
                [3, 112, 672, 1, True,  HS],
                [5, 160, 672, 2, True,  HS],
                [5, 160, 960, 1, True,  HS],
                [5, 160, 960, 1, True,  HS],
            ]
        elif model_size == 'Small':
            last_channel = 576, 1024
            inverted_residual_setting = [
                #k, out, exp, s, se, nl
                [3, 16, 16,  2, True,  RE],
                [3, 24, 72,  2, False, RE],
                [3, 24, 88,  1, False, RE],
                [5, 40, 96,  2, True,  HS],
                [5, 40, 240, 1, True,  HS],
                [5, 40, 240, 1, True,  HS],
                [5, 48, 120, 1, True,  HS],
                [5, 48, 144, 1, True,  HS],
                [5, 96, 288, 2, True,  HS],
                [5, 96, 576, 1, True,  HS],
                [5, 96, 576, 1, True,  HS]
            ]
        else:
            raise NotImplementedError

        # building first layer
        # in_channels = _make_divisible(in_channels * width_mult, 8)
        features = [ConvBN(3, in_channels, kernel_size=1, stride=2,activation=HS)]

        # building inverted residual blocks
        for k, out_channels, exp, s, se, nl in inverted_residual_setting:
            features.append(InvertedResidual(in_channels, out_channels, exp, k, s, se, nl))
            in_channels = out_channels

        # building last several layers
        features.append(ConvBN(in_channels, last_channel[0], kernel_size=1, activation=HS))

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Linear(last_channel[0], last_channel[1]),
            HS,
            nn.Dropout(0.8),
            nn.Linear(last_channel[1], n_class)
        )
        # weight initialization
        self.initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)