# This is file for construting the NN models

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import scipy.fftpack as sft

# For the parameters selection, I inherit the choice of PINO

def compl_mul2d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixy,ioxy->boxy", a, b)

class DCNN2d(nn.Module):
    def __init__(self, modes1, modes2, width=64, fc_dim=128, layers=None, in_dim=3, out_dim=1, activation='tanh', device=None):
        super(DCNN2d, self).__init__()

        # PINO implements the four Fourier layers, but here I would like to try with Four Cosine layers

        """
        Input: the solution of the coefficients and x grids and y grids (a, x, y)
        Output: the computed solution
        """
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2d(in_size, out_size, mode1_num, mode2_num, device) for in_size, out_size, mode1_num, mode2_num in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1) for in_size, out_size in zip(self.layers, self.layers[1:])])
    
        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)

        if activation =='tanh':
            self.activation = F.tanh
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation == F.relu
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        '''
        Args:
            - x : (batch size, x_grid, y_grid, 2)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        # print(length - 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            # print(i)
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y)
            # print(type(x1))
            # print(type(x2))
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x



# 2D Cosine Transformation Layer
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, device):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Cosine modes to multiply, at most floor(N/2) + 1

        # ???
        # Cannot understand why only the modes2 is restrictde to floor(N/2) + 1

        self.modes1 = modes1
        self.modes2 = modes2

        # Here I use the PINO approach, but I use the float not cfloat
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.float))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.float))

        self.device = device

    def forward(self, x):
        batchsize = x.shape[0]
        # Apply Discrete Cosine Transformation 
        x_dct = torch.from_numpy(sft.dctn(x.detach().cpu().numpy(), type=1, axes=[2, 3])).float().to(self.device)

        # Multiply relevant Cosine modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = compl_mul2d(x_dct[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = compl_mul2d(x_dct[:, :, -self.modes1:, :self.modes2], self.weights2)
        # Apply invese  Discrete Cosine Transformation
        x = torch.from_numpy(sft.idctn(out_ft.detach().cpu().numpy(), type=1, shape=(x.size(-2), x.size(-1)), axes=[2, 3])).float().to(self.device)

        # print(x)
        # print(type(x))
        return x
    
