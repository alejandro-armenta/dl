import torch

import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):

        super(ResidualBlock, self).__init__()
            
        self.conv1 = nn.Sequential(
            
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, 
                      kernel_size=3, 
                      stride=stride, 
                      padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()

        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3, 
                      stride=1, 
                      padding=1
            ),
            nn.BatchNorm2d(out_channels)
        )
        
        self.downsample = downsample
        self.relu = nn.ReLU()

        self.out_channels = out_channels

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        
        residual = x
        if self.downsample:
            residual = self.downsample(residual)
        
        out += residual
        
        out = self.relu(out)

        return out

#ResidualBlock(3,3)


class ResNet(nn.Module):

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes,kernel_size=1,stride=stride),
                nn.BatchNorm2d(planes)
            )

        layers = []
        
        layers.append(
            block(in_channels=self.inplanes, out_channels=planes, stride=stride, downsample=downsample)
        )
        
        for i in range(1, blocks):
            layers.append(
                block(in_channels=self.inplanes, out_channels=planes)
            )
        
        return nn.Sequential(*layers)

    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        
        self.inplanes = 64
        self.make_layer(block=block, planes=64, blocks=layers[0])
        #self.make_layer(layers[1])
        #self.make_layer(layers[2])
        #self.make_layer(layers[3])
        
        nn.Linear(in_features=512,out_features=num_classes)


ResNet(ResidualBlock, [3,4,6,3])
