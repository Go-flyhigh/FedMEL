import torch.nn.functional as F
from torch import nn

from experiments.FLHetero.Blocks import ConvBlock, LinearBlock
def calc_feat_size(input_size: int) -> int:
    """
    conv1(5x5, stride=1) -> pool(2x2) -> conv2(5x5, stride=1) -> pool(2x2)
    CIFAR: input=32 -> 5
    Tiny-ImageNet: input=64 -> 13
    """
    size = input_size
    size = (size - 4) // 2
    size = (size - 4) // 2
    return size

class CNN_1_large(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10, c_expand=8, input_size=32):
        super(CNN_1_large, self).__init__()
        self.input_size = input_size

        self.Block1 = ConvBlock(
            in_channels=in_channels,
            expand_channels=8,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5)
        feat_size = calc_feat_size(input_size)
        self.fc1 = nn.Linear(2* n_kernels * feat_size * feat_size, 2000)
        self.fc2 = nn.Linear(2000, 500)

        self.Block2 = LinearBlock(in_features=500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x, m_rep_large=None):
        x = self.Block1(x) + x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        hetero_rep = self.fc2(x)
        delta = self.Block2(hetero_rep)
        hetero_rep = F.relu(hetero_rep + delta)

        x = self.fc3(hetero_rep)
        return x, hetero_rep

class CNN_2_large(nn.Module): # change filters of convs
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10, c_expand=8, input_size=32):
        super(CNN_2_large, self).__init__()
        self.input_size = input_size

        self.Block1 = ConvBlock(
            in_channels=in_channels,
            expand_channels=8,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, n_kernels, 5)
        feat_size = calc_feat_size(input_size)
        self.fc1 = nn.Linear(n_kernels * feat_size * feat_size, 2000)
        self.fc2 = nn.Linear(2000, 500)

        self.Block2 = LinearBlock(in_features=500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x, m_rep_large=None):
        x = self.Block1(x) + x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        hetero_rep = self.fc2(x)

        delta = self.Block2(hetero_rep)
        hetero_rep = F.relu(hetero_rep + delta)

        x = self.fc3(hetero_rep)
        return x, hetero_rep

class CNN_3_large(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10, c_expand=8, input_size=32):
        super(CNN_3_large, self).__init__()
        self.input_size = input_size

        self.Block1 = ConvBlock(
            in_channels=in_channels,
            expand_channels=8,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5)
        feat_size = calc_feat_size(input_size)
        self.fc1 = nn.Linear(2* n_kernels * feat_size * feat_size, 1000)
        self.fc2 = nn.Linear(1000, 500)

        self.Block2 = LinearBlock(in_features=500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x, m_rep_large=None):
        x = self.Block1(x) + x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        hetero_rep = self.fc2(x)

        delta = self.Block2(hetero_rep)
        hetero_rep = F.relu(hetero_rep + delta)

        x = self.fc3(hetero_rep)
        return x, hetero_rep


class CNN_4_large(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10, c_expand=8, input_size=32):
        super(CNN_4_large, self).__init__()
        self.input_size = input_size

        self.Block1 = ConvBlock(
            in_channels=in_channels,
            expand_channels=8,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5)
        feat_size = calc_feat_size(input_size)
        self.fc1 = nn.Linear(2* n_kernels * feat_size * feat_size, 800)
        self.fc2 = nn.Linear(800, 500)

        self.Block2 = LinearBlock(in_features=500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x, m_rep_large=None):
        x = self.Block1(x) + x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        hetero_rep = self.fc2(x)

        delta = self.Block2(hetero_rep)
        hetero_rep = F.relu(hetero_rep + delta)

        x = self.fc3(hetero_rep)
        return x, hetero_rep

class CNN_5_large(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10, c_expand=8, input_size=32):
        super(CNN_5_large, self).__init__()
        self.input_size = input_size
        self.Block1 = ConvBlock(
            in_channels=in_channels,
            expand_channels=8,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5)
        feat_size = calc_feat_size(input_size)
        self.fc1 = nn.Linear(2* n_kernels * feat_size * feat_size, 500)
        self.fc2 = nn.Linear(500, 500)
        self.Block2 = LinearBlock(in_features=500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x, m_rep_large=None):
        x = self.Block1(x) + x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        hetero_rep = self.fc2(x)

        delta = self.Block2(hetero_rep)
        hetero_rep = F.relu(hetero_rep + delta)

        x = self.fc3(hetero_rep)
        return x, hetero_rep

