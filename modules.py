import torch
import torch.nn as nn
import torch.nn.functional as F

#双卷积模块： Conv -> ReLU -> Conv -> ReLU
class DoubleConv(nn.Module):
    def __init__ (self,in_ch,out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
        
#下采样模块: MaxPool -> DoubleConv
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.pool_conv(x)
    
#上采样模块：Upsamle -> DoubleConv
class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            #双线性上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        else:
            #反卷积上采样
            self.up = nn.ConvTranspose2d(in_ch, in_ch //2,kernel_size=2,stride=2)

        
        #上采样后通道数拼接，所以输入通道数为in_ch
        self.conv = DoubleConv(in_ch, out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        #对齐大小，防止奇数尺寸带来的不匹配
        diffy = x2.size()[2] - x1.size()[2]
        diffx = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffx // 2,diffx - diffx // 2,
                        diffy // 2, diffy - diffy // 2])
        
        #拼接通道
        x = torch.cat([x2,x1], dim=1)
        return self.conv(x)
    
#最后一层 1*1卷积，将特征图映射到输出通道
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#基础U-NET
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1,bilinear=True):
        
        """
        Args:
            in_channels (int):输入通道数, RGB图像通常为3
            out_channels (int): 输出通道数, 二分类分割通常为1
            features (list):每一层的特征图数量
            bilinear (bool) :是否使用双线性插值进行上采样
        """
        super(UNet, self).__init__()
        self.bilinear = bilinear

        #编码器
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        factor = 2 if bilinear else 1 
        self.down4 = Down(512,1024 // factor)

        #解码器
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        #输出层
        self.outc = OutConv(64,out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits






        