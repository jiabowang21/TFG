import torch.nn as nn

class NeuralNetworkMF(nn.Module):
    # Red neuronal para tratar múltiplos fotogramas
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(782, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.linear(x)