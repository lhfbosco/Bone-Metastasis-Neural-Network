import torch.nn as nn


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(4, 4, 4)),
            nn.Conv2d(25, 25, kernel_size=(3, 3, 3)),
            nn.Conv2d(25, 25, kernel_size=(3, 3, 3), dilation=(1, 2, 2)),
            nn.Conv2d(25, 25, kernel_size=(3, 3, 3), dilation=(2, 4, 4)),
            nn.MaxUnpool2d(25, 25, kernel_size=(4, 4, 4))
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(4, 4, 4)),
            nn.Conv2d(25, 25, kernel_size=(3, 3, 3)),
            nn.Conv2d(25, 25, kernel_size=(3, 3, 3), dilation=(1, 2, 2)),
            nn.Conv2d(25, 25, kernel_size=(3, 3, 3), dilation=(2, 4, 4)),
            nn.MaxUnpool2d(25, 25, kernel_size=(2, 2, 2))
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(3, 25, kernel_size=(3, 3, 3)),
            nn.Conv2d(25, 25, kernel_size=(3, 3, 3), dilation=(1, 2, 2)),
            nn.Conv2d(25, 25, kernel_size=(3, 3, 3), dilation=(2, 4, 4))
        )
        self.layer4 = nn.Sequential(
           nn.Conv2d(75, 75, kernel_size=(3, 3, 3), dilation=(2, 4, 4)),
           nn.Conv2d(75, 100, kernel_size=(3, 3, 3), dilation=(2, 4, 4)),
           nn.Conv2d(100, 100, kernel_size=(3, 3, 3), dilation=(2, 4, 4)),
           nn.Conv2d(100, 100, kernel_size=(3, 3, 3), dilation=(2, 4, 4)),
           nn.Conv2d(100, 2, kernel_size=(1, 1, 1))
        )

    def forward(self,x):
        out1 = self.layer1(x)
        out2 = self.layer2(x)
        out3 = self.layer3(x)
        out = self.layer4(out1, out2, out3)
        return out
