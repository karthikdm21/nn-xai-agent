import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Fully connected
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 32x32 -> 16x16
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 16x16 -> 8x8
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # 8x8 -> 4x4
        x = x.view(x.size(0), -1)                          # flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = CNN()
    dummy = torch.randn(1, 3, 32, 32)
    out = model(dummy)
    print("Model output shape:", out.shape)  # should be [1, 10]
    print("Model architecture:\n", model)