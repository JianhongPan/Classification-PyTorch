from torch import nn
from torch.nn import ReLU6 as RE
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, activation=None):
        padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            activation if activation is not None else nn.Sequential()
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        mid_channels = in_channels * expand_ratio
        self.is_shortcut = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers += [
                ConvBN(in_channels, mid_channels, kernel_size=1, activation=RE(inplace=True))
            ]
        layers += [
            #dw
            ConvBN(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=mid_channels, activation=RE(inplace=True)),
            ConvBN(mid_channels, out_channels, kernel_size=1)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.is_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.0, num_classes=1000):
        """
        MobileNet V2

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount

        """
        super(MobileNetV2, self).__init__()
        in_channels = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        in_channels = _make_divisible(in_channels * width_mult, 8)
        features = [ConvBN(3, in_channels, kernel_size=3, stride=2, activation=RE(inplace=True))]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            out_channels = _make_divisible(c * width_mult, 8)
            stride = s
            for i in range(n):
                features.append(InvertedResidual(in_channels, out_channels, stride, expand_ratio=t))
                in_channels = out_channels
                stride = 1

        # building last several layers
        out_channels = _make_divisible(last_channel * max(1.0, width_mult), 8)
        features.append(ConvBN(in_channels, out_channels, kernel_size=1, activation=RE(inplace=True)))
        in_channels = out_channels

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_channels, num_classes),
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