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
    pass

