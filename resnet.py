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
        
        #skip connection

        residual = self.conv1(x)
        residual = self.conv2(residual)

        if self.downsample:
            x_ = self.downsample(x)
        
        result = x_ + residual
        
        result = self.relu(result)

        return result

#ResidualBlock(3,3)


class ResNet(nn.Module):

    def make_layer(self, block, planes, blocks, stride=1):
        
        downsample = None

        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes)
            )

        layers = []
        
        layers.append(
            block(in_channels=self.inplanes, out_channels=planes, stride=stride, downsample=downsample)
        )
        
        self.inplanes = planes

        for i in range(1, blocks):
            layers.append(
                block(in_channels=self.inplanes, out_channels=planes)
            )
        
        return nn.Sequential(*layers)

    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        
        self.inplanes = 64

        self.conv1 = nn.Sequential(
            #esto realmente significa que entran a color y salen en 64
            #entra a color y salen 64 channels 
            

            #batch, in_channels, 224, 224
            
            #batch, out_channels, 112, 112

            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),

            nn.BatchNorm2d(64),

            nn.ReLU()

        )

        #batch, 64, 112, 112

        #batch, 64, 56, 56

        #picks the most prominent feature
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self.make_layer(block=block, planes=64, blocks=layers[0], stride=1)

        self.layer1 = self.make_layer(block=block, planes=128, blocks=layers[1], stride=2)
        
        self.layer2 = self.make_layer(block=block, planes=256, blocks=layers[2], stride=2)
        
        self.layer3 = self.make_layer(block=block, planes=512, blocks=layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

        self.fc = nn.Linear(in_features=512,out_features=num_classes)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x = self.layer0(x)

        #aqui vas
        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)

        #flatten, batch size
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


#a = ResNet(ResidualBlock, [3,4,6,3])


