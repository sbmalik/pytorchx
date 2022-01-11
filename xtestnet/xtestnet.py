import torch
import torch.nn as nn
from torchsummary import summary


class XTestNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.cb1 = self._conv_block(3, 16, 5)
        self.cb2 = self._conv_block(16, 32, 5)
        self.res1 = self._conv_block(32, 64, 3, padding=1)
        self.res2 = self._conv_block(64, 64, 3, padding=1)
        self.res_same = self._conv_block(32, 64, 1)
        self.cb3 = self._conv_block(64, 128, 5)
        self.cb4 = self._conv_block(128, 8, 13)
        self.fc = self._classifier()

    def _classifier(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 4 * 4, 10),
            nn.Softmax(dim=-1)
        )

    def _conv_block(self, in_c, out_c, ks, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, (ks, ks), padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU())

    def _residual_block(self, x):
        res = self.res1(x)
        res = self.res2(res)
        x = self.res_same(x)
        return x + res

    def forward(self, ips):
        x = self.cb1(ips)
        x = self.cb2(x)
        x = self._residual_block(x)
        x = self.cb3(x)
        x = self.cb4(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    net = XTestNet()
    net = net.to('cuda:0')
    inputs = torch.ones(1, 3, 28, 28, dtype=torch.float32).to('cuda:0')
    # print(summary(net, (3, 28, 28)))
    # net = net.to('cuda:0')
    net.eval()
    with torch.no_grad():
        res = net(inputs)
    torch.save(net, 'xtestnet.pth')
    print(res)
