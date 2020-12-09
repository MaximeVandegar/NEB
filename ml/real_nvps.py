import torch
import torch.nn as nn
import math
import sys

sys.path.append(".")
from utils import GaussianDistribution


class RealNVP(nn.Module):
    """
    RealNVP defined in "Density estimation using Real NVP,
                        Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio,
                        2017"
    """

    def __init__(self, data_dim, context_dim, hidden_layer_dim, alpha=1.9):
        super().__init__()

        self.d = data_dim // 2
        output_dim = data_dim - self.d
        self.alpha = alpha  # alpha for the clamping operator defined in "Guided Image Generation
                            #                                             with Conditional Invertible
                            #                                             Neural Networks"

        self.s = nn.Sequential(
            nn.Linear(self.d + context_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, output_dim)
        )

        self.t = nn.Sequential(
            nn.Linear(self.d + context_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, output_dim)
        )

    def forward(self, y, context=None):
        return self.forward_and_compute_log_jacobian(y, context)[0]

    def forward_and_compute_log_jacobian(self, y, context=None):
        """
        Computes z = f(y) where f is bijective, and the log jacobian determinant of the transformation

        Arguments:
        ----------
            - y: a batch of input variables
            - context: conditioning variables (optional)

        Return:
        -------
            - z = f(y)
            - log_jac
        """

        z1 = y[:, :self.d]

        s = self.s(torch.cat((z1, context), dim=1)) if (context is not None) else self.s(z1)
        s = self.clamp(s)

        t = self.t(torch.cat((z1, context), dim=1)) if (context is not None) else self.t(z1)
        z2 = torch.mul(y[:, self.d:], torch.exp(s)) + t

        return torch.cat((z1, z2), dim=1), torch.sum(s, dim=1)

    def invert(self, z, context=None):
        """
        Computes y = f^{-1}(z)

        Arguments:
        ----------
            - z: a batch of input variables
            - context: conditioning variables (optional)
        """

        y1 = z[:, :self.d]

        s = self.s(torch.cat((y1, context), dim=1)) if (context is not None) else self.s(y1)
        s = self.clamp(s)

        t = self.t(torch.cat((y1, context), dim=1)) if (context is not None) else self.t(y1)
        y2 = torch.mul(z[:, self.d:] - t, torch.exp(-s))

        return torch.cat((y1, y2), dim=1)

    def clamp(self, s):
        """
        Clamping operator defined in "Guided Image Generation with Conditional Invertible Neural Networks
                                      Lynton Ardizzone, Carsten Lüth, Jakob Kruse, Carsten Rother, Ullrich Köthe
                                      2019"
        """
        return torch.tensor([2 * self.alpha / math.pi], device=s.device) * torch.atan(s / self.alpha)


class RealNVPs(nn.Module):
    """
    Stacking RealNVP layers. The same (almost) API than UMNN (https://github.com/AWehenkel/UMNN) is provided.
                             API: - forward(y, context=None)
                                  - invert(y, context=None)
                                  - compute_ll(y, context=None)
    """

    def __init__(self, flow_length, data_dim, context_dim, hidden_layer_dim, alpha=1.9):
        super().__init__()

        self.data_dim = data_dim
        self.layers = nn.Sequential(
            *(RealNVP(data_dim, context_dim, hidden_layer_dim, alpha=alpha) for _ in range(flow_length)))

    def compute_ll(self, y, context=None):
        z, log_jacobian = self.forward_and_compute_log_jacobian(y, context)
        log_y_likelihood = GaussianDistribution.log_pdf(z) + log_jacobian

        return log_y_likelihood, z

    def forward(self, y, context=None):
        return self.forward_and_compute_log_jacobian(y, context)[0]

    def forward_and_compute_log_jacobian(self, y, context=None):

        log_jacobians = 0
        z = y
        for layer in self.layers:
            # Every iteration, we swap the first and last variables so that no variable is left unchanged
            z = torch.cat((z[:, self.data_dim // 2:], z[:, :self.data_dim // 2]), dim=1)

            z, log_jacobian = layer.forward_and_compute_log_jacobian(z, context)
            log_jacobians += log_jacobian

        return z, log_jacobians

    def invert(self, z, context=None):

        y = z
        for i in range(len(self.layers) - 1, -1, -1):
            y = self.layers[i].invert(y, context)

            # Every iteration, we swap the first and last variables so that no variable is left unchanged
            y = torch.cat((y[:, (self.data_dim - self.data_dim // 2):], y[:, :(self.data_dim - self.data_dim // 2)]),
                          dim=1)

        return y


def test_real_nvp_layer(data_dim=4, number_data=10000):
    nvp_layer = RealNVP(data_dim, 0, 50)

    x = torch.rand(number_data, data_dim) * 10
    error = (x - nvp_layer.invert(nvp_layer(x)[0])).abs().sum() / number_data
    print('Test 1, error:', error)

    error = (x - nvp_layer(nvp_layer.invert(x))[0]).abs().sum() / number_data
    print('Test 2, error:', error)


def test_real_nvp_layers(data_dim=4, number_data=10000):
    nvp_layer = RealNVPs(6, data_dim, 0, 50)

    x = torch.rand(number_data, data_dim) * 10
    error = (x - nvp_layer.invert(nvp_layer(x)[0])).abs().sum() / number_data
    print('Test 1, error:', error)

    error = (x - nvp_layer(nvp_layer.invert(x))[0]).abs().sum() / number_data
    print('Test 2, error:', error)
