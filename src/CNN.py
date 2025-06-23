import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: додати Batch Normalization

class Encoder3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=256):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 128, kernel_size=(3, 4, 4), padding=1)
        self.bn1 = nn.BatchNorm3d(128)
        self.conv2 = nn.Conv3d(128, 192, kernel_size=(3, 5, 5), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(192)
        self.conv3 = nn.Conv3d(192, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

class Decoder3D(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.deconv1 = nn.ConvTranspose3d(in_channels, 192, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(192)
        self.deconv2 = nn.ConvTranspose3d(192, 128, kernel_size=(3, 5, 5), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(128)
        self.deconv3 = nn.ConvTranspose3d(128, out_channels, kernel_size=(3, 4, 4), padding=1)
        self.upsample = nn.Upsample(scale_factor=(1.0, 2.0, 2.0), mode='trilinear', align_corners=False)


    def forward(self, x):
        x = self.upsample(F.relu(self.bn1(self.deconv1(x))))
        x = self.upsample(F.relu(self.bn2(self.deconv2(x))))
        x = torch.sigmoid(self.deconv3(x))
        return x


class ClassifierHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_channels, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.pool(x).view(x.size(0), -1)  # Flatten
        return self.fc(x)

class SemiSupervised3DCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.encoder = Encoder3D(in_channels, 256)
        self.decoder = Decoder3D(256, in_channels)
        self.classifier = ClassifierHead(256, num_classes)

    def forward(self, x, mode='classify'):
        latent = self.encoder(x)
        if mode == 'reconstruct':
            return self.decoder(latent)
        elif mode == 'classify':
            return self.classifier(latent)
        elif mode == 'full':
            return self.classifier(latent), self.decoder(latent)
        else:
            raise ValueError("mode must be one of: 'full', 'classify', 'reconstruct'")