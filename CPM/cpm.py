import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    def __init__(self, in_c, outc):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, outc, kernel_size=7, padding=3)

    def forward(self, x):
        return self.conv(x)


class Stage(nn.Module):
    def __init__(self, in_c, outc):
        """
        CNNs for inference at Stage t (t>=2)
        :param outc:
        """
        super(Stage, self).__init__()
        self.Mconv1 = ConvBlock(in_c, 128)
        self.Mconv2 = ConvBlock(128, 128)
        self.Mconv3 = ConvBlock(128, 128)
        self.Mconv4 = ConvBlock(128, 128)
        self.Mconv5 = ConvBlock(128, 128)
        self.Mconv6 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv7 = nn.Conv2d(128, outc, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.Mconv1(x))
        x = self.relu(self.Mconv2(x))
        x = self.relu(self.Mconv3(x))
        x = self.relu(self.Mconv4(x))
        x = self.relu(self.Mconv5(x))
        x = self.relu(self.Mconv6(x))
        x = self.Mconv7(x)
        return x


class Stage1(nn.Module):
    def __init__(self, inc, outc):
        super(Stage1, self).__init__()
        self.stage1_1 = nn.Conv2d(inc, 512, kernel_size=1, padding=0)
        self.stage1_2 = nn.Conv2d(512, outc, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: feature map   4D Tensor   batch size * 128 * 46 * 46
        :return: x              4D Tensor   batch size * 21  * 46 * 46
        """
        x = self.relu(self.stage1_1(x))  # batch size * 512 * 46 * 46
        x = self.stage1_2(x)             # batch size * 21 * 46 * 46
        return x


class VGG19(nn.Module):
    def __init__(self, pretrained=False):
        super(VGG19, self).__init__()
        self.backbone = models.vgg19(pretrained=pretrained).features[:27]

        self.relu = nn.ReLU(inplace=True)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 128, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(self.conv5_1(x))  # 512
        x = self.relu(self.conv5_2(x))  # 512
        x = self.relu(self.conv5_3(x))  # 128
        return x  # Batchsize * 128 * H/8 * W/8


class CPMHand(nn.Module):
    def __init__(self, outc, pretrained=False):
        super(CPMHand, self).__init__()
        self.outc = outc                       # 21
        self.vgg19 = VGG19(pretrained=pretrained)                    # backbone
        self.stage1 = Stage1(128, self.outc)
        self.stage2 = Stage(self.outc + 128, self.outc)
        self.stage3 = Stage(self.outc + 128, self.outc)
        self.stage4 = Stage(self.outc + 128, self.outc)
        self.stage5 = Stage(self.outc + 128, self.outc)
        self.stage6 = Stage(self.outc + 128, self.outc)

    def forward(self, image):
        """
        :param image: (FloatTensor) size:(B,3,368,368)
        :return: 
        """
        # ************* backbone *************
        features = self.vgg19(image)        # batch size * 128 * 46 * 46

        # ************* CM stage *************
        stage1 = self.stage1(features)   # batch size * 21 * 46 * 46
        stage2 = self.stage2(torch.cat([features, stage1], dim=1))
        stage3 = self.stage3(torch.cat([features, stage2], dim=1))
        stage4 = self.stage4(torch.cat([features, stage3], dim=1))
        stage5 = self.stage5(torch.cat([features, stage4], dim=1))
        stage6 = self.stage6(torch.cat([features, stage5], dim=1))

        return torch.stack([stage1, stage2, stage3, stage4, stage5, stage6], dim=1)


if __name__ == "__main__":
    net = CPMHand(21)
    x = torch.randn(2, 3, 368, 368)
    if torch.cuda.is_available():
        x = x.cuda()
        net = net.cuda()
    y = net(x)            # (2, 6, 21, 46, 46)
    print(y.shape)


