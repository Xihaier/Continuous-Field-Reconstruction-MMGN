import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from functools import partial
from torch.nn import init
from torch.nn.parameter import Parameter


#---------------------------------------------------------
# Swish function
#---------------------------------------------------------
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


#---------------------------------------------------------
# non-linear activation set
#---------------------------------------------------------
nl_s = {'relu': partial(nn.ReLU),
       'sigmoid': partial(nn.Sigmoid),
       'tanh': partial(nn.Tanh),
       'selu': partial(nn.SELU),
       'softplus': partial(nn.Softplus),
       'gelu': partial(nn.GELU),
       'swish': partial(Swish),
       'elu': partial(nn.ELU)}


#---------------------------------------------------------
# initialization methods set
#---------------------------------------------------------
init_s = {"uniform": partial(torch.nn.init.uniform_),
          "normal": partial(torch.nn.init.normal_),
          "ones": partial(torch.nn.init.ones_),
          "zeros": partial(torch.nn.init.zeros_),
          "xavier_u": partial(torch.nn.init.xavier_uniform_),
          "xavier_n": partial(torch.nn.init.xavier_normal_),
          "kaiming_u": partial(torch.nn.init.kaiming_uniform_),
          "kaiming_n": partial(torch.nn.init.kaiming_normal_),
          "trunc_n": partial(torch.nn.init.trunc_normal_),
          "orthogonal": partial(torch.nn.init.orthogonal_)}


#---------------------------------------------------------
# linear transformation fusing coordinates and latent information
#---------------------------------------------------------
class FuseLinear(nn.Module):
    def __init__(self, d_coord: int, d_latent: int, d_out: int) -> None:
        super(FuseLinear, self).__init__()

        self.b    = d_coord
        self.A    = Parameter(torch.empty(d_out, d_coord))
        self.B    = Parameter(torch.empty(d_out, d_latent))
        self.bias = Parameter(torch.empty(d_out))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.b)
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.B, a=math.sqrt(5))
        init.uniform_(self.bias, -bound, bound)

    def forward(self, coord: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        # coord: b, d_points, d_coord
        # latent: b, d_latent
        # out: b, d_points, d_out
        linear_coord  = torch.einsum('bpi,oi->bpo', coord, self.A)
        linear_latent = torch.einsum('bj,oj->bo', latent, self.B)
        linear_latent = linear_latent.unsqueeze(1)

        out = 0
        out += linear_coord 
        out += linear_latent
        out += self.bias
        return out


#---------------------------------------------------------
# Linear layer
#---------------------------------------------------------
class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        return

    def forward(self, x):
        return self.linear(x)


#---------------------------------------------------------
# Fourier layer
#---------------------------------------------------------
class FourierLayer(nn.Module):
    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.weight = Parameter(torch.empty((out_features, in_features)))
        self.weight_scale = weight_scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # x = coord: b, d_points, d_coord
        # out: b, d_points, d_out * 2
        return torch.cat([torch.sin(F.linear(x, self.weight * self.weight_scale)), torch.cos(F.linear(x, self.weight * self.weight_scale))], dim=-1)


#---------------------------------------------------------
# Gabor layer
#---------------------------------------------------------
class GaborLayer(nn.Module):
    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :])


#---------------------------------------------------------
# Multiplicative and Modulated Gabor Network
#---------------------------------------------------------
class MMGNet(nn.Module):
    def __init__(
            self, 
            in_size: int = 2,
            hidden_size: int = 300,
            n_data: int = 2,
            latent_size: int = 370,
            latent_init: str = "zeros",
            out_size: int = 1,
            n_layers: int = 7, 
            input_scale: int = 370, 
            alpha: float = 1.0,
            filter: str = "Gabor",
            ):
        super(MMGNet, self).__init__()

        # model parameters
        self.bilinear = nn.ModuleList(
            [FuseLinear(in_size, latent_size, hidden_size)] +
            [FuseLinear(hidden_size, latent_size, hidden_size) for _ in range(int(n_layers))]
        )
        if filter == "Gabor":
            self.filters = nn.ModuleList(
                [GaborLayer(in_size, hidden_size, input_scale/np.sqrt(n_layers+1), alpha/(n_layers+1)) for _ in range(n_layers + 1)]
            )  
        elif filter == "Fourier":
            self.filters = nn.ModuleList(
                [FourierLayer(in_size, hidden_size // 2, input_scale / np.sqrt(n_layers + 1)) for _ in range(n_layers + 1)]
            )
        elif filter == "Linear":
            self.filters = nn.ModuleList(
                [LinearLayer(in_size, hidden_size) for _ in range(n_layers + 1)]
            )    
        self.output_layer = nn.Linear(hidden_size, out_size)

        # latent variables
        self.latents = nn.Parameter(torch.FloatTensor(n_data, latent_size))
        self.latents = init_s[latent_init](self.latents)

    def forward(self, x, idx):
        latent = self.latents[idx,:]
        zi = self.filters[0](x) * self.bilinear[0](x*0., latent)
        for i in range(1, len(self.filters)):
            zi = self.bilinear[i](zi, latent) * self.filters[i](x)
        
        return self.output_layer(zi)


if __name__ == "__main__":
    model = MMGNet(
                    in_size = 2,
                    hidden_size = 300,
                    n_data = 5,
                    latent_size = 370,
                    latent_init = "zeros",
                    out_size = 1,
                    n_layers = 7, 
                    input_scale = 370, 
                    alpha = 1.0,
                    filter = "Linear",
                   )
    input_coord = torch.rand((5, 700, 2))
    input_idx = [0, 1, 2, 3 ,4]
    output_y = model(input_coord, input_idx)
    print(output_y.shape)