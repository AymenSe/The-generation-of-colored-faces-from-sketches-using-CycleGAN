import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs) if down 
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1, padding_mode='reflect'),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1, padding_mode='reflect')
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()

        # self.initial = nn.Conv2d(img_channels, num_features, kernel_size=7, padding=3, padding_mode="reflect")

        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        self.down_blocks = nn.ModuleList(
            [
             ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1,  padding_mode='reflect'),
             ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1,  padding_mode='reflect'),
            ]
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.ModuleList(
            [
             ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
             ConvBlock(num_features*2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
            ]
        )

        self.last = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3,  padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
          x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
          x = layer(x)
        return torch.tanh(self.last(x))