import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(34, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.linear(x)