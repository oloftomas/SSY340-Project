import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):

    def unet_conv(self, in_channels, out_channels, leaky):
        if leaky:
            # use leaky relu to avoid vanishing gradients
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

    def up_sampl(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, 1),
            nn.ReLU()
        )

    def __init__(self, leaky):

        super(UNet, self).__init__()

        # Four encoding layers
        self.conv1 = self.unet_conv(1, 64, leaky)
        self.conv2 = self.unet_conv(64, 128, leaky)
        self.conv3 = self.unet_conv(128, 256, leaky)
        self.conv4 = self.unet_conv(256, 512, leaky)
        self.conv5 = self.unet_conv(512, 1024, leaky)

        # pooling
        self.pool = nn.MaxPool2d(2)

        # Four upsampling layers
        self.up1 = self.up_sampl(1024, 512)
        self.up2 = self.up_sampl(512, 256)
        self.up3 = self.up_sampl(256, 128)
        self.up4 = self.up_sampl(128, 64)

        # Four decoding layers
        self.conv6 = self.unet_conv(1024, 512, False)
        self.conv7 = self.unet_conv(512, 256, False)
        self.conv8 = self.unet_conv(256, 128, False)
        self.conv9 = self.unet_conv(128, 64, False)

        # Last layer
        self.conv10 = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        # Encoding path
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))

        # Decoding path
        x = self.conv6(torch.cat((x4, self.up1(x5)), 1))
        x = self.conv7(torch.cat((x3, self.up2(x)), 1))
        x = self.conv8(torch.cat((x2, self.up3(x)), 1))
        x = self.conv9(torch.cat((x1, self.up4(x)), 1))
        x = self.conv10(x)
        m = nn.Tanh()
        x = m(x)
        
        return x

class DNet(nn.Module):
    def dnet_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def __init__(self):
        super(DNet, self).__init__()

        # Five conv layers
        self.conv1 = self.dnet_conv(3,64)
        self.conv2 = self.dnet_conv(64,128)
        self.conv3 = self.dnet_conv(128,256)
        self.conv4 = self.dnet_conv(256,512)
        self.conv5 = self.dnet_conv(512,1024)

        # Pooling layer
        self.pool = nn.MaxPool2d(2)

        # Last layer
        self.conv6 = nn.Linear(2*2*1024, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))
        
        x6 = x5.view(-1, 2 * 2 * 1024)
        m = nn.Sigmoid()
        x = m(self.conv6(x6))

        return x
