import numpy as np
import torch
from tqdm import tqdm
from inference.neb.util import create_dataloader
import sys
sys.path.append(".")
from utils import GaussianDistribution


class IwEstimator:
    @staticmethod
    def compute_estimator(minibatch, device, source_data_dim, prior_network,
                          recognition_network, log_likelihood_fct, K):
        """
        Estimate the log marginal likelihood of the minibatch with the variational estimator
        defined in: 'Importance Weighted Autoencoders,
                     Yuri Burda, Roger Grosse, Ruslan Salakhutdinov,
                     2015'
        """

        # Format the minibatch for efficient computation of the marginal likelihood on GPU
        dataset = torch.zeros(K, minibatch.shape[0], minibatch.shape[1], device=device)
        dataset += minibatch.to(device)
        dataset = torch.transpose(dataset, 0, 1)
        minibatch = dataset.reshape(minibatch.shape[0] * K, minibatch.shape[1])

        # Samples from the recognition network
        z = GaussianDistribution.sample(minibatch.shape[0], dim=source_data_dim).to(device)
        x, log_jacobian = recognition_network.forward_and_compute_log_jacobian(z, minibatch.to(device))

        log_posterior = GaussianDistribution.log_pdf(z) - log_jacobian
        log_y_likelihood = log_likelihood_fct(minibatch.to(device), x)
        log_prior, _ = prior_network.compute_ll(x)

        # Estimates the log marginal likelihood
        kl_divergence = log_posterior - log_prior
        importance_densities = (log_y_likelihood - kl_divergence).view(-1, K)
        log_marginal_likelihood = torch.logsumexp(importance_densities, dim=1).sum()

        return log_marginal_likelihood

    @staticmethod
    def infer(observations, prior_network, optimizer_prior_network, recognition_network, optimizer_recognition_network,
              log_likelihood_fct, source_data_dim, K=128, nb_epochs=300, batch_size=128, validation_set_size=0.1,
              early_stopping=True, early_stopping_patience=10, early_stopping_verbose=True):

        assert batch_size <= observations.shape[0], 'The number of observations should be greater or equal than the ' \
                                                    'batch size '

        validation_set_size = int(observations.shape[0] * validation_set_size)
        validation_set = observations[:validation_set_size]
        observations = observations[validation_set_size:]
        early_stopping_count = 0
        validation_loss = []
        best_validation_loss = float('inf')

        training_loss = []
        for epoch in tqdm(range(nb_epochs)):

            # Training
            batch_loss = []
            dataloader = create_dataloader(observations, batch_size)
            for batch in dataloader:
                log_marginal_likelihood = IwEstimator.compute_estimator(batch, batch.device, source_data_dim,
                                                                        prior_network, recognition_network,
                                                                        log_likelihood_fct, K)

                optimizer_prior_network.zero_grad()
                optimizer_recognition_network.zero_grad()
                loss = - log_marginal_likelihood
                loss.backward()
                optimizer_prior_network.step()
                optimizer_recognition_network.step()

                batch_loss.append(loss.item())

            training_loss.append(np.mean(batch_loss))

            # Validate
            prior_network.eval()
            recognition_network.eval()

            batch_loss = []
            dataloader = create_dataloader(validation_set, batch_size)
            for batch in dataloader:
                log_marginal_likelihood = IwEstimator.compute_estimator(batch, batch.device, source_data_dim,
                                                                        prior_network, recognition_network,
                                                                        log_likelihood_fct, K)
                loss = - log_marginal_likelihood
                batch_loss.append(loss.item())

            validation_loss.append(np.mean(batch_loss))

            prior_network.train()
            recognition_network.train()

            if validation_loss[-1] < best_validation_loss:
                early_stopping_count = 0
                best_validation_loss = validation_loss[-1]
            else:
                early_stopping_count += 1
                if early_stopping and (early_stopping_count > early_stopping_patience):
                    if early_stopping_verbose:
                        print(
                            f'Validation loss did not improve for {early_stopping_patience} epochs, stopping training '
                            f'after {epoch} epochs.')
                    break

        return training_loss, validation_loss
