import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=9):
        super(SpatialAttention, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        pad = (self.kernel_size - 1) // 2  # Padding on one side for stride 1

        self.grp1_conv1k = nn.Conv2d(self.in_channels,
                                     self.in_channels // 2,
                                     (1, self.kernel_size),
                                     padding=(0, pad))
        self.grp1_bn1 = nn.BatchNorm2d(self.in_channels // 2)
        self.grp1_convk1 = nn.Conv2d(self.in_channels // 2,
                                     1, (self.kernel_size, 1),
                                     padding=(pad, 0))
        self.grp1_bn2 = nn.BatchNorm2d(1)

        self.grp2_convk1 = nn.Conv2d(self.in_channels,
                                     self.in_channels // 2,
                                     (self.kernel_size, 1),
                                     padding=(pad, 0))
        self.grp2_bn1 = nn.BatchNorm2d(self.in_channels // 2)
        self.grp2_conv1k = nn.Conv2d(self.in_channels // 2,
                                     1, (1, self.kernel_size),
                                     padding=(0, pad))
        self.grp2_bn2 = nn.BatchNorm2d(1)

    def forward(self, input_):
        # Generate Group 1 Features
        grp1_feats = self.grp1_conv1k(input_)
        grp1_feats = F.relu(self.grp1_bn1(grp1_feats))
        grp1_feats = self.grp1_convk1(grp1_feats)
        grp1_feats = F.relu(self.grp1_bn2(grp1_feats))

        # Generate Group 2 features
        grp2_feats = self.grp2_convk1(input_)
        grp2_feats = F.relu(self.grp2_bn1(grp2_feats))
        grp2_feats = self.grp2_conv1k(grp2_feats)
        grp2_feats = F.relu(self.grp2_bn2(grp2_feats))

        added_feats = torch.sigmoid(torch.add(grp1_feats, grp2_feats))
        added_feats = added_feats.expand_as(input_).clone()

        return added_feats


class Model(nn.Module):
    r"""TAS

    Args:
        num_class (int): Number of classes for the classification task
        backbone (str): 'resnet34'
        pooling (bool): pooling before mlp or not
        normalize (bool): normalize features or not
        activation (str): apply activation function for output or not
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, (T_{in}), C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, num_class)` 
          
    """
    def __init__(self,
                 num_class,
                 backbone='resnet34',
                 pooling=True,
                 normalize=True,
                 activation='',
                 **kwargs):
        super().__init__()
        self.num_class = num_class
        self.backbone = backbone

        self.pooling = pooling
        self.normalize = normalize
        self.activation = activation

        if self.backbone == 'resnet34':
            self.encoder = nn.Sequential(
                *list(models.resnet34(pretrained=False).children())
                [:-2],  # [N, 512, image_size // (2^4), _]
            )
            self.output_channel = 512
            self.output_size = 8

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.pooling == False:
            self.spa_attn = nn.ModuleList([
                SpatialAttention(self.output_channel, kernel_size=3)
                for _ in range(num_class)
            ])
            if self.activation == 'sigmoid':
                self.final = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(
                            self.output_channel * self.output_size *
                            self.output_size, 64),
                        nn.LeakyReLU(inplace=True),
                        nn.Linear(64, 1),
                        nn.Sigmoid(),
                    ) for _ in range(num_class)
                ])
            else:
                self.final = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(
                        self.output_channel * self.output_size *
                        self.output_size, 64),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(64, 1),
                ) for _ in range(num_class)
            ])
        else:
            self.spa_attn = nn.ModuleList([
                SpatialAttention(self.output_channel, kernel_size=3)
                for _ in range(num_class)
            ])
            if self.activation == 'sigmoid':
                self.final = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.output_channel, 64),
                        nn.LeakyReLU(inplace=True),
                        nn.Linear(64, 1),
                        nn.Sigmoid(),
                    ) for _ in range(num_class)
                ])
            else:
                self.final = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.output_channel, 64),
                        nn.LeakyReLU(inplace=True),
                        nn.Linear(64, 1),
                    ) for _ in range(num_class)
                ])

    def forward(self, image):
        
        if len(image.shape) < 5:
            N, _, _, _ = image.shape
            T = 1
            x = image
        else:
            N, T, _, _, _ = image.shape
            x = image.view(-1, image.shape[2], image.shape[3], image.shape[4])
        x = self.encoder(x)

        features = []
        feat_w_attns = []
        attn_weights = []
        for idx in range(self.num_class):
            attn_weight = self.spa_attn[idx](x)
            feat_w_attn = torch.mul(x, attn_weight)

            if self.pooling == False:
                feat_w_attn = feat_w_attn.view(feat_w_attn.shape[0], -1)
            else:
                feat_w_attn = self.avgpool(feat_w_attn)
                feat_w_attn = feat_w_attn.view(feat_w_attn.shape[0], -1)

            if self.normalize:
                feat_w_attn = F.normalize(feat_w_attn, p=2, dim=1)

            features.append(feat_w_attn)
            feat_w_attns.append(feat_w_attn)
            attn_weights.append(attn_weight[:, 0, :, :])

        x = torch.stack(feat_w_attns, dim=-1)  # [N*T, D, num_class]
        feature = torch.stack(features, dim=-1)
        attn_weights = torch.stack(attn_weights, dim=-1)

        cls_outputs = []
        for idx in range(self.num_class):
            cls_outputs.append(self.final[idx](x[:, :, idx]))
        output = torch.stack(cls_outputs, dim=-1).squeeze(1)

        return output, feature, attn_weights