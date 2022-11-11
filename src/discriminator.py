import torch.nn as nn

# discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.main = nn.Sequential(
            *self.block(in_channels, 64, norm=False),  # 28-14
            *self.block(64, 128, norm=True),  # 141-7
            nn.ZeroPad2d((1, 0, 1, 0)),  # 7-8
            nn.Conv2d(128, 1, 4, 1, 1))  # 8-7  fianl shape (batch_size x1x7x7)

    @staticmethod
    def block(in_channels, out_channels, norm=False):
        block = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
        if norm:
            block.append(nn.InstanceNorm2d(out_channels))
        block.append(nn.LeakyReLU(0.2, inplace=True))
        return block

    def forward(self, x):
        return self.main(x)
