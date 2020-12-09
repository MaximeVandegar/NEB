import torch
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
import math


class TwoMoonsSimulator:
    """
    Simulator defined in: Automatic Posterior Transformation for Likelihood-free Inference
                          David S. Greenberg, Marcel Nonnenmacher, Jakob H. Macke
                          17 May 2019
    """

    def __init__(self, mean_radius=0.1, std_radius=0.01):
        self.mean_radius = mean_radius
        self.std_radius = std_radius
        self.ydim = 2
        self.xdim = 2

    def get_ground_truth_parameters(self):
        return torch.tensor([0., 0.7])

    def sample_prior(self, size):
        m = Uniform(torch.zeros((size, 2)) - 1., torch.zeros((size, 2)) + 1.)
        theta = m.sample()
        x = theta
        return x

    def corrupt(self, x):
        theta = x
        size_theta = theta.shape

        uniform = Uniform(torch.zeros(size_theta[0]) + torch.tensor([-math.pi / 2]),
                          torch.zeros(size_theta[0]) + torch.tensor([math.pi / 2]))
        normal = Normal(torch.zeros(size_theta[0]) + self.mean_radius, torch.zeros(size_theta[0]) + self.std_radius)

        a = uniform.sample()
        r = normal.sample()

        p = torch.empty(size_theta)

        p[:, 0] = torch.mul(r, torch.cos(a)) + 0.25
        p[:, 1] = torch.mul(r, torch.sin(a))

        q = torch.empty(size_theta)
        q[:, 0] = -torch.abs(theta[:, 0] + theta[:, 1]) / math.sqrt(2)
        q[:, 1] = (-theta[:, 0] + theta[:, 1]) / math.sqrt(2)

        x = p + q

        y = x
        return y
