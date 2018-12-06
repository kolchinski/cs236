import torch

class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.reshape(x.size(0), *self.shape)

class Generator(torch.nn.Module):
    def __init__(self, dim_z=64, num_channels=1):
        super().__init__()
        self.dim_z = dim_z
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_z, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(512, 64 * 7 * 7),
            torch.nn.BatchNorm1d(64 * 7 * 7),
            torch.nn.ReLU(inplace=True),
            Reshape(64, 7, 7),

            torch.nn.PixelShuffle(2),
            torch.nn.Conv2d(64 // 4, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),

            torch.nn.PixelShuffle(2),
            torch.nn.Conv2d(32 // 4, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(torch.nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),

            Reshape(64 * 7 * 7),
            torch.nn.Linear(64 * 7 * 7, 512),
            torch.nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Linear(512, 1),
            Reshape()
        )

    def forward(self, x):
        return self.net(x)

class ConditionalGenerator(torch.nn.Module):
    def __init__(self, dim_z=64, num_channels=1, num_classes=10):
        super().__init__()
        self.dim_z = dim_z
        self.num_classes = num_classes
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_z + num_classes, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(512, 64 * 7 * 7),
            torch.nn.BatchNorm1d(64 * 7 * 7),
            torch.nn.ReLU(inplace=True),
            Reshape(64, 7, 7),

            torch.nn.PixelShuffle(2),
            torch.nn.Conv2d(64 // 4, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),

            torch.nn.PixelShuffle(2),
            torch.nn.Conv2d(32 // 4, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, z, y):
        one_hot_y = torch.eye(self.num_classes, device=y.device)[y]
        z = torch.cat([z, one_hot_y], 1)
        return self.net(z)

class ConditionalDiscriminator(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1),

            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1),

            Reshape(64 * 7 * 7),
            torch.nn.Linear(64 * 7 * 7, 512),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(512, self.num_classes)
        )

    def forward(self, x, y):
        return self.net(x).gather(1, y.unsqueeze(1)).squeeze(1)
        