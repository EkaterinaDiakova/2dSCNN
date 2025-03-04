import torch
import torch.nn as nn
import torch.nn.functional as F
import math as m

class SpikeFunctionBoxcar(torch.autograd.Function):
    """
    Compute surrogate gradient of the spike step function using
    box-car function similar to DECOLLE, Kaiser et al. (2020).
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        grad_x[x <= -0.5] = 0
        grad_x[x > 0.5] = 0
        return grad_x

class SpikeCNN(nn.Module):
    def __init__(self, frames, bands, n_classes, dropout_rate=0.3):
        super(SpikeCNN, self).__init__()

        # Initialize a ModuleList to hold all layers
        self.snn = nn.ModuleList()
        self.frames = frames
        self.bands = bands
        self.n_classes = n_classes
        self.is_snn = False

        self.snn.append(FeatureEncoder())
        self.snn.append(SConv2d(in_channels=1, out_channels=8, kernel_size=(6, 6), padding='same', bias=False))
        self.snn.append(SAvgPool2d(kernel_size=(2, 2)))
        self.snn.append(SConv2d(in_channels=8, out_channels=8, kernel_size=(6, 6), padding='same', bias=False))
        self.snn.append(SAvgPool2d(kernel_size=(1, 2)))
        self.snn.append(SConv2d(in_channels=8, out_channels=8, kernel_size=(6, 6), padding='same', bias=False))

        self.snn.append(FlattenAndPermuteLayer())

        self.snn.append(IFLayer(input_size=(8 * (frames // 2) * (bands // 4)), hidden_size=32, dropout = dropout_rate))
        self.snn.append(IFLayer(input_size=32, hidden_size=n_classes, dropout = dropout_rate))
        self.snn.append(Readout(n_classes,n_classes,dropout_rate))

    def forward(self, x):
        for lay in self.snn:
            x = lay(x)
        return x, None, None

class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        self.time_steps = 8

    def normalize_matrix(self, matrix):
        matrix = matrix.squeeze()
        min_val = matrix.amin(dim=(1, 2), keepdim=True)
        max_val = matrix.amax(dim=(1, 2), keepdim=True)
        normalized_matrix = self.time_steps * (matrix - min_val) / (max_val - min_val)

        return normalized_matrix.unsqueeze(1)

    def create_binary_sequences(self, normalized_matrix):
        batch_size, _, frames, bands = normalized_matrix.shape

        three_d_matrix = torch.zeros((batch_size, frames, bands, self.time_steps), dtype=torch.float)
        current_values = normalized_matrix.squeeze(1).clone()

        for t in range(self.time_steps):
            mask = current_values > 0
            three_d_matrix[:, :, :, t] = mask.float()
            current_values[mask] -= 1

        shuffled_indices = torch.randperm(self.time_steps)
        shuffled_three_d_matrix = three_d_matrix[:, :, :, shuffled_indices]

        return shuffled_three_d_matrix.unsqueeze(1)

    def forward(self, feature_matrix):
      normalized_matrix = self.normalize_matrix(feature_matrix)
      return self.create_binary_sequences(normalized_matrix).to(feature_matrix.device)

class SConv2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=8, kernel_size=(6, 6), padding='same', bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.thresh = nn.Parameter(torch.tensor([0.8]), requires_grad=True)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


    def forward(self, x):
        batch_size, _, num_windows, num_features, num_spikes = x.shape
        print(x.shape)
        device = x.device

        ut = torch.zeros(batch_size, self.out_channels, num_windows, num_features).to(device)
        st = torch.zeros(batch_size, self.out_channels, num_windows, num_features).to(device)

        s = []

        for t in range(num_spikes):
            Wx = self.conv(x[:, :, :, :, t])
            ut = ut - st * self.thresh + Wx

            st = SpikeFunctionBoxcar.apply(ut)

            s.append(st)
        return torch.stack(s, dim=4)


class SAvgPool2d(nn.Module):
    def __init__(self, kernel_size=(2, 2)):
        super(SAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.thresh = nn.Parameter(torch.tensor([0.8]), requires_grad=True)

    def forward(self, x):
        batch_size, out_channels, num_windows, num_features, num_spikes = x.shape
        
        x_reshaped = x.permute(0, 1, 4, 2, 3).reshape(batch_size * out_channels * num_spikes, 1, num_windows, num_features)
        
        pooled = nn.functional.avg_pool2d(x_reshaped, self.kernel_size)

        pooled_shape = pooled.shape
        pooled = pooled.view(batch_size, out_channels, num_spikes, pooled_shape[2], pooled_shape[3])
        
        pooled = pooled.permute(0, 1, 3, 4, 2)
        
        device = x.device

        ut = torch.zeros(batch_size, out_channels, pooled.shape[2], pooled.shape[3]).to(device)
        st = torch.zeros(batch_size, out_channels, pooled.shape[2], pooled.shape[3]).to(device)

        s = []

        for t in range(num_spikes):
            ut = ut - st * self.thresh + pooled[:, :, :, :, t]

            st = SpikeFunctionBoxcar.apply(ut)

            s.append(st)

        return torch.stack(s, dim=4)
        
class FlattenAndPermuteLayer(nn.Module):
    def __init__(self):
        super(FlattenAndPermuteLayer, self).__init__()
        self.flatten = nn.Flatten(end_dim=-2)

    def forward(self, x):
        x = self.flatten(x)
        x = x.permute(0, 2, 1)
        return x


class IFLayer(nn.Module):

    def __init__(self,input_size, hidden_size, dropout):
        super().__init__()
        # Trainable parameters
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.thresh = nn.Parameter(torch.tensor([0.8]), requires_grad=True)
        self.weight = nn.Parameter(torch.rand(self.hidden_size, self.input_size))

        nn.init.uniform_(self.weight, -1/(m.sqrt(self.input_size)), 1/(m.sqrt(self.input_size)))
                
        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        Wx = nn.functional.linear(x, self.weight)
        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)
        s = self.drop(s)

        return s

    def _lif_cell(self, Wx):
        device = Wx.device

        ut = (torch.ones(Wx.shape[0], Wx.shape[2])).to(device) * self.thresh / 2
        st = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device)

        s = []

        # Loop over time axis
        for t in range(Wx.shape[1]):
            ut = ut - st*self.thresh + Wx[:, t, :]
            # Compute spikes with surrogate gradient
            st = SpikeFunctionBoxcar.apply(ut - self.thresh)
            s.append(st)
        return torch.stack(s, dim=1)


class Readout(nn.Module):
    def __init__(
        self, input_size, hidden_size, dropout):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        # Trainable parameters
        # self.weight = nn.Parameter(torch.rand(self.hidden_size, self.input_size))
        # nn.init.uniform_(self.weight, -1/(m.sqrt(self.input_size)), 1/(m.sqrt(self.input_size)))

        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Wx = F.linear(x, self.weight)
        out = self._readout_cell(x)

        return out

    def _readout_cell(self, Wx):

        device = Wx.device
        y = 0
        for t in range(Wx.shape[1]):
            y += F.softmax(Wx[:, t, :], dim=-1)
        return y
