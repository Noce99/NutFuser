import os
import torch
from torch import nn

class simple_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def compute_loss(self, predicted, real):
        loss = self.loss_function(predicted, real)
        return loss