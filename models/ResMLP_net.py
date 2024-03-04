import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial


#---------------------------------------------------------
# Nonlinear activation
#---------------------------------------------------------
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


non_act = {'relu': partial(nn.ReLU),
       'sigmoid': partial(nn.Sigmoid),
       'tanh': partial(nn.Tanh),
       'selu': partial(nn.SELU),
       'softplus': partial(nn.Softplus),
       'gelu': partial(nn.GELU),
       'swish': partial(Swish),
       'elu': partial(nn.ELU)}


#---------------------------------------------------------
# Residual block
#---------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, width, act, use_batchnorm=True, use_skipconnect=True):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(width, width, bias=False)
        self.fc2 = nn.Linear(width, width, bias=False)
        self.act1 = non_act[act]()
        self.act2 = non_act[act]()
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(width)
            self.bn2 = nn.BatchNorm1d(width)
        self.use_skipconnect = use_skipconnect

    def forward(self, x):
        # x shape: (batch_size, width)
        identity = x

        # conv -> bn -> non-act
        out = self.fc1(x)
        if self.use_batchnorm:
            out = torch.permute(out, (0, 2, 1))
            out = self.bn1(out)
            out = torch.permute(out, (0, 2, 1))
        out = self.act1(out)
        
        # conv -> bn -> non-act
        out = self.fc2(out)
        if self.use_batchnorm:
            out = torch.permute(out, (0, 2, 1))
            out = self.bn2(out)
            out = torch.permute(out, (0, 2, 1))
        out = self.act2(out)
        
        # skip connection
        if self.use_skipconnect:
            return out + identity
        else:
            return out


#---------------------------------------------------------
# ResMLP
#---------------------------------------------------------
class ResMLP(nn.Module):
    def __init__(self, 
        res_in_dim: int = 0,
        res_out_dim: int = 0,
        res_width: int = 0,
        res_depth: int = 0,
        net_act: str = None,
        ):
        super(ResMLP, self).__init__()

        self.fc_in = nn.Linear(res_in_dim, res_width)
        self.act_in = non_act[net_act]()
        self.fcs = nn.ModuleList([ResBlock(res_width, net_act) for i in range(res_depth)])
        self.fc_out = nn.Linear(res_width, res_out_dim)

    def forward(self, coord):
        # 1st layer
        out = self.act_in(self.fc_in(coord))

        # res blocks
        for fc in self.fcs:        
            out = fc(out)
        
        # last layer
        out = self.fc_out(out)
        return out


if __name__ == "__main__":
    model = ResMLP(
                res_in_dim = 3,
                res_out_dim = 1,
                res_width = 128,
                res_depth = 4,
                net_act = "swish",
                   )
    input_x = torch.rand((1, 700, 3))
    output_y = model(input_x)
    print(output_y.shape)