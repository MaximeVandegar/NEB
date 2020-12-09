import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


class InverseKinematicsSimulator:
    """
    Simulator defined in: "Analyzing Inverse Problems with Invertible Neural Networks"
                           Ardizzone et al.
                           Feb 2019
                           
    Minor modification: stochastic noise is added to the rotating joints.
    """

    def __init__(self, prior_var=None, l1=1 / 2, l2=1 / 2, l3=1, noise=0.01 / 180 * np.pi):
        if prior_var is None:
            prior_var = [1 / 16, 1 / 4, 1 / 4, 1 / 4]
        self.prior_var = torch.tensor(prior_var)
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.noise = noise
        self.ydim = 2
        self.xdim = 4

    def get_ground_truth_parameters(self):
        return torch.tensor([0.1, -0.4, 0.5, -0.1])

    def sample_prior(self, size):
        prior = MultivariateNormal(torch.zeros(size, 4), self.prior_var * torch.eye(4))
        return prior.sample()

    def corrupt(self, x):
        y = torch.empty((x.shape[0], 2))

        noise_distribution = Normal(torch.zeros(x.shape[0]), self.noise)

        y[:, 0] = self.l1 * torch.sin(x[:, 1] + noise_distribution.sample()) + self.l2 * torch.sin(
            x[:, 2] + x[:, 1] + noise_distribution.sample()) + self.l3 * torch.sin(
            x[:, 3] + x[:, 1] + x[:, 2] + noise_distribution.sample()) + x[:, 0]
        y[:, 1] = self.l1 * torch.cos(x[:, 1] + noise_distribution.sample()) + self.l2 * torch.cos(
            x[:, 2] + x[:, 1] + noise_distribution.sample()) + self.l3 * torch.cos(
            x[:, 3] + x[:, 1] + x[:, 2] + noise_distribution.sample())

        return y
