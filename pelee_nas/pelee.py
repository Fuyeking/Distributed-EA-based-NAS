import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math


class _DenseLayer(nn.Module):
    # 对每个dense_block的dense_way的多少和卷积核多少也进行变化
    def __init__(self, num_input_features, growth_rate, bottleneck_width, drop_rate, dense_way, con_num,
                 intricate=False):
        super(_DenseLayer, self).__init__()
        self.intricate = intricate  # 进行复杂的演化
        self.dense_way = dense_way
        if not self.intricate:
            growth_rate = int(growth_rate / 2)
            inter_channel = int(growth_rate * bottleneck_width / 4) * 4
            '''
            if inter_channel > num_input_features / 2:
                inter_channel = int(num_input_features / 8) * 4
                print('adjust inter_channel to ', inter_channel)
            '''
            self.branch1a = BasicConv2d(num_input_features, inter_channel, kernel_size=1)
            self.branch1b = BasicConv2d(inter_channel, growth_rate, kernel_size=3, padding=1)

            self.branch2a = BasicConv2d(num_input_features, inter_channel, kernel_size=1)
            self.branch2b = BasicConv2d(inter_channel, growth_rate, kernel_size=3, padding=1)
            self.branch2c = BasicConv2d(growth_rate, growth_rate, kernel_size=3, padding=1)
        else:
            growth_rate = int(growth_rate / dense_way)
            inter_channel = int(growth_rate * bottleneck_width / 4) * 4
            if inter_channel > num_input_features / 2:
                inter_channel = int(num_input_features / 8) * 4
                print('adjust inter_channel to ', inter_channel)
            if dense_way == 1:
                self._dense_one_way(num_input_features, growth_rate, inter_channel)
            elif dense_way == 2:
                self._dense_two_way(num_input_features, growth_rate, inter_channel)
            else:
                pass

    def _dense_one_way(self, num_input_features, growth_rate, inter_channel):
        self.branch1a = BasicConv2d(num_input_features, inter_channel, kernel_size=1)
        self.branch1b = BasicConv2d(inter_channel, growth_rate, kernel_size=3, padding=1)

    def _dense_two_way(self, num_input_features, growth_rate, inter_channel):
        self.branch1a = BasicConv2d(num_input_features, inter_channel, kernel_size=1)
        self.branch1b = BasicConv2d(inter_channel, growth_rate, kernel_size=3, padding=1)

        self.branch2a = BasicConv2d(num_input_features, inter_channel, kernel_size=1)
        self.branch2b = BasicConv2d(inter_channel, growth_rate, kernel_size=3, padding=1)
        self.branch2c = BasicConv2d(growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        if not self.intricate:
            branch1 = self.branch1a(x)
            branch1 = self.branch1b(branch1)

            branch2 = self.branch2a(x)
            branch2 = self.branch2b(branch2)
            branch2 = self.branch2c(branch2)
            block = torch.cat([x, branch1, branch2], 1)
        else:
            if self.dense_way == 1:
                branch1 = self.branch1a(x)
                branch1 = self.branch1b(branch1)
                block = torch.cat([x, branch1], 1)
            elif self.dense_way == 2:
                branch1 = self.branch1a(x)
                branch1 = self.branch1b(branch1)

                branch2 = self.branch2a(x)
                branch2 = self.branch2b(branch2)
                branch2 = self.branch2c(branch2)
                block = torch.cat([x, branch1, branch2], 1)
            else:
                block = x
        return block


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, dense_way=None,
                 con_num=None):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            intricate = False
            if dense_way is not None:
                intricate = True
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate,
                                dense_way=dense_way, con_num=con_num, intricate=intricate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _StemBlock(nn.Module):
    def __init__(self, num_input_channels, num_init_features):
        super(_StemBlock, self).__init__()

        num_stem_features = int(num_init_features / 2)

        self.stem1 = BasicConv2d(num_input_channels, num_init_features, kernel_size=3, stride=2, padding=1)
        self.stem2a = BasicConv2d(num_init_features, num_stem_features, kernel_size=1, stride=1, padding=0)
        self.stem2b = BasicConv2d(num_stem_features, num_init_features, kernel_size=3, stride=2, padding=1)
        self.stem3 = BasicConv2d(2 * num_init_features, num_init_features, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.stem1(x)

        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)
        branch1 = self.pool(out)

        out = torch.cat([branch1, branch2], 1)
        out = self.stem3(out)

        return out


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, activation=True, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


class PeleeNet(nn.Module):
    r"""PeleeNet model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf> and
     "Pelee: A Real-Time Object Detection System on Mobile Devices" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int or list of 4 ints) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bottleneck_width (int or list of 4 ints) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    # 只演化growth_rate,和每个block的layer数量
    def __init__(self, growth_rate=[8, 8, 8, 8], block_config=[3, 4, 6, 8], dense_nums=None, con_nums=[3, 3, 3, 3],
                 num_init_features=32, bottleneck_width=[1, 2, 4, 4], drop_rate=0.05, num_classes=10):
        super(PeleeNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('stemblock', _StemBlock(3, num_init_features)),
        ]))

        if type(growth_rate) is list:
            growth_rates = growth_rate
            assert len(growth_rates) == 4, 'The growth rate must be the list and the size must be 4'
        else:
            growth_rates = [growth_rate] * 4

        if type(bottleneck_width) is list:
            bottleneck_widths = bottleneck_width
            assert len(bottleneck_widths) == 4, 'The bottleneck width must be the list and the size must be 4'
        else:
            bottleneck_widths = [bottleneck_width] * 4

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if dense_nums is not None:#演化dense_way的数量
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bottleneck_widths[i], growth_rate=growth_rates[i], drop_rate=drop_rate,
                                    dense_way=dense_nums[i], con_num=con_nums[i])
            else:  # dense_way和num_layers为空，只演化每个block的growth_rate 和 layer 数量
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bottleneck_widths[i], growth_rate=growth_rates[i], drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rates[i]

            self.features.add_module('transition%d' % (i + 1), BasicConv2d(
                num_features, num_features, kernel_size=1, stride=1, padding=0))

            if i != len(block_config) - 1:
                self.features.add_module('transition%d_pool' % (i + 1), nn.AvgPool2d(kernel_size=2, stride=2))
                num_features = num_features

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.drop_rate = drop_rate
        self._initialize_weights()

    def forward(self, x):
        features = self.features(x)
        out = F.avg_pool2d(features, kernel_size=(features.size(2), features.size(3))).view(features.size(0), -1)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.classifier(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
