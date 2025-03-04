import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class CNN(nn.Module):
    def __init__(self, frames, bands, n_classes, dropout_rate=0.3):
        super(CNN, self).__init__()

        # Initialize a ModuleList to hold all layers
        self.ann = nn.ModuleList()
        self.frames = frames
        self.bands = bands
        self.n_classes = n_classes
        self.is_snn = False

        self.ann.append(ClipFloor_enc())

        self.ann.append(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(6, 6), padding='same', bias=False))
        self.ann.append(ClipFloor())
        self.ann.append(nn.AvgPool2d(kernel_size=(2, 2)))
        self.ann.append(ClipFloor())

        self.ann.append(nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(6, 6), padding='same', bias=False))
        self.ann.append(ClipFloor())
        self.ann.append(nn.AvgPool2d(kernel_size=(1, 2)))
        self.ann.append(ClipFloor())

        self.ann.append(nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(6, 6), padding='same', bias=False))
        self.ann.append(ClipFloor())

        self.ann.append(nn.Flatten())

        self.ann.append(nn.Linear(8 * (frames // 2) * (bands // 4), 32, bias=False))
        self.ann.append(ClipFloor())
        self.ann.append(nn.Dropout(dropout_rate))

        self.ann.append(nn.Linear(32, n_classes, bias=False))
        self.ann.append(ClipFloor())
        self.ann.append(nn.Dropout(dropout_rate))

        self.ann.append(nn.Softmax(dim=1))

    def forward(self, x):
        # Reshape input to (batch_size, channels, height, width)
        x = x.view(-1, 1, self.frames, self.bands)
        for ann_lay in self.ann:
            x = ann_lay(x)

        return x, None, None

class ClipFloor(nn.Module):
#ClipFloor activation function from https://arxiv.org/pdf/2303.04347

    def __init__(self):

        super().__init__()
        self.thresh = nn.Parameter(torch.tensor([0.8]), requires_grad=True)
        self.L = 8
        self.clip = GradFloor.apply

    def forward(self, x):

        x = x / self.thresh
        x = torch.clamp(x, 0, 1)
        x = self.clip(x * self.L + 0.5) / self.L
        #x = x * self.thresh#
        return x

class ClipFloor_enc(nn.Module):
#ClipFloor encoding layer. Using max value of feature matrix to initialise threshold.
    def __init__(self):
        super().__init__()
        self.L = 8
        self.clip = GradFloor.apply

    def forward(self, x):

        matrix = x.squeeze()
        thresh = torch.max(matrix.amax(dim=(1, 2), keepdim=True))
        x = (matrix / thresh).unsqueeze(1)
        x = torch.clamp(x, 0, 1)
        x = self.clip(x * self.L + 0.5) / self.L

        return x

