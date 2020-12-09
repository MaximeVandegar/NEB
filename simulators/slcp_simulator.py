import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform


class SLCPSimulator:
    """
    Simulator defined in: Sequential Neural Likelihood; Fast Likelihood-free Inference 
                          with Autoregressive Flows
                          George Papamakarios, David C. Sterratt, Iain Murray
                          21 Jan 2019
    """

    def __init__(self):
        self.ydim = 8
        self.xdim = 5

    def get_ground_truth_parameters(self):
        return torch.tensor([-0.7, -2.9, -1.0, -0.9, 0.6])

    def sample_prior(self, size):
        uniform = Uniform(torch.zeros(size, 5) + torch.tensor([-3.]), torch.zeros(size, 5) + torch.tensor([3.]))
        return uniform.sample()

    def corrupt(self, x):
        means = x[:, :2]
        s1 = torch.pow(x[:, 2], 2)
        s2 = torch.pow(x[:, 3], 2)
        pho = torch.tanh(x[:, 4])

        cov = torch.zeros(x.shape[0], 2, 2)
        cov[:, 0, 0] = torch.pow(s1, 2)
        cov[:, 0, 1] = pho * s1 * s2
        cov[:, 1, 0] = pho * s1 * s2
        cov[:, 1, 1] = torch.pow(s2, 2)

        normal = MultivariateNormal(means, cov)

        y = torch.zeros(x.shape[0], 8)
        y[:, :2] = normal.sample()
        y[:, 2:4] = normal.sample()
        y[:, 4:6] = normal.sample()
        y[:, 6:] = normal.sample()

        return y
