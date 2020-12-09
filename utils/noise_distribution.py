import math
import torch
from torch.distributions import MultivariateNormal


class GaussianDistribution:

    @staticmethod
    def sample(size, dim=2):
        means = torch.zeros(size, dim)
        cov = torch.eye(dim)
        m = MultivariateNormal(means, cov)
        return m.sample()

    @staticmethod
    def log_pdf(z):
        """
        Arguments:
        ----------
            - z: a batch of m data points (size: m x data_dim)
        """
        return -.5 * (torch.log(torch.tensor([math.pi * 2], device=z.device)) + z ** 2).sum(1)
