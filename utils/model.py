import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out


class LinearDiffusion(nn.Module):
    def __init__(self, n_steps, y_dim=10, fp_dim=128, feature_dim=512, guidance=False):
        super(LinearDiffusion, self).__init__()
        n_steps = n_steps + 1
        self.y_dim = y_dim
        self.guidance = guidance
        self.feature_dim = feature_dim
        self.norm = nn.BatchNorm1d(feature_dim)

        # Unet
        if self.guidance:
            self.lin1 = ConditionalLinear(y_dim + fp_dim, feature_dim, n_steps)
        else:
            self.lin1 = ConditionalLinear(y_dim, feature_dim, n_steps)

        self.unetnorm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm1d(feature_dim)
        self.lin4 = nn.Linear(feature_dim, feature_dim)
        self.lin5 = nn.Linear(feature_dim, y_dim)

    def forward(self, y, timesteps, fp_x=None, x_embed=None):

        if x_embed is not None:
            x_embed = self.norm(x_embed)
        else:
            x_embed = torch.ones([y.shape[0], self.feature_dim])

        if self.guidance:
            y = torch.cat([y, fp_x], dim=-1)

        y = self.lin1(y, timesteps)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        y = x_embed * y
        y = self.lin2(y, timesteps)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        y = self.lin3(y, timesteps)
        y = self.unetnorm3(y)
        y = F.relu(y)
        y = self.lin4(y)
        y = F.relu(y)
        y = self.lin5(y)
        return y