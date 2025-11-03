import torch
import torch.nn as nn
import torch.nn.functional as F


class MusicCNN(nn.Module):
    """
    Simple CNN model for music genre classification.
    Uses mel-spectrogram as input.
    """

    def __init__(self, num_classes=10, input_channels=1):
        """
        Args:
            num_classes: Number of genre classes
            input_channels: Number of input channels (1 for mono mel-spectrogram)
        """
        super(MusicCNN, self).__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # Calculate output size after conv layers
        # Input: 128x128 -> After 4 maxpool(2,2): 8x8
        self.fc_input_size = 256 * 8 * 8

        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, 1, height, width]

        Returns:
            logits: Output tensor [batch_size, num_classes]
        """
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class ImprovedMusicCNN(nn.Module):
    """
    Improved CNN with residual connections and attention.
    """

    def __init__(self, num_classes=10, input_channels=1):
        super(ImprovedMusicCNN, self).__init__()

        # Initial conv
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Residual blocks
        self.res_block1 = self._make_res_block(64, 128)
        self.res_block2 = self._make_res_block(128, 256)
        self.res_block3 = self._make_res_block(256, 512)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _make_res_block(self, in_channels, out_channels):
        """Create residual block"""
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        identity = self.skip(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)

        return out


class ResNetMusic(nn.Module):
    """
    ResNet-style architecture for music classification.
    More powerful but requires more data.
    """

    def __init__(self, num_classes=10, input_channels=1):
        super(ResNetMusic, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        if stride != 1:
            layers.append(nn.MaxPool2d(2, 2))

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
