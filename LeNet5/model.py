import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.features = nn.Sequential(
            # C1
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),  # S2

            # C3
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),  # S4

            # C5
            nn.Conv2d(16, 120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(120, 84),  # F6
            nn.Tanh(),
            RBFOutput(in_features=84, num_classes=n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

class LeNet5_ReLU(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.features = nn.Sequential(
            # C1
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),  # S2

            # C3
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),  # S4

            # C5
            nn.Conv2d(16, 120, kernel_size=5, stride=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(120, 84),  # F6
            nn.ReLU(),
            nn.Linear(84, n_classes)  # Output
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class LeNet5_224(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            # C1
            nn.Conv2d(3, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),   # S2

            # C3
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),   # S4

            # C5
            nn.Conv2d(64, 128, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),   # S6

            # C7 (마지막 Conv)
            nn.Conv2d(128, 120, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(120 * 22 * 22, 84),  # F6
            nn.ReLU(),
            nn.Linear(84, n_classes)       # Output
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # Flatten
        x = self.classifier(x)
        return x


class RBFOutput(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        # class prototypes: shape (num_classes, in_features)
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_features))

    def forward(self, x):
        # x shape: (batch, in_features)
        # compute squared euclidean distance to each prototype
        # => output shape: (batch, num_classes)
        # o_k = - || x - w_k ||^2
        x = x.unsqueeze(1)                    # (B, 1, 84)
        w = self.prototypes.unsqueeze(0)      # (1, C, 84)
        dist = torch.sum((x - w)**2, dim=2)   # (B, C)
        return -dist