import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, num_classess = 10):
        super(NeuralNet, self).__init__()
        # First hidden layer:
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Second hidden layer:
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output layer, fc stan for fully connected:
        self.fc = nn.Linear(7*7*32, num_classess)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)  # Convert to 1D np array
        out = self.fc(out) 
        return out