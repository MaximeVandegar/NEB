import torch
import numpy as np
from tqdm import tqdm
from inference.neb.util import create_dataloader
import sys
sys.path.append(".")
from utils import GaussianDistribution


class McBiasedEstimator:

    @staticmethod
    def compute_estimator(minibatch, nb_mc_integration_steps, device, source_data_dim, prior_network,
                          log_likelihood_fct):
        """
        Estimate the log marginal likelihood of the minibatch with Monte Carlo integration (biased)
        """

        # Formats the minibatch for efficient computation on GPU
        dataset = torch.zeros(nb_mc_integration_steps, minibatch.shape[0], minibatch.shape[1], device=device)
        dataset += minibatch.to(device)
        dataset = torch.transpose(dataset, 0, 1)
        minibatch = dataset.reshape(minibatch.shape[0] * nb_mc_integration_steps, minibatch.shape[1])

        # Samples from the prior
        z = GaussianDistribution.sample(minibatch.shape[0], dim=source_data_dim).to(device)
        x = prior_network(z)

        # Estimates the log marginal likelihood
        log_y_likelihoods = log_likelihood_fct(minibatch.to(device), x)
        log_y_likelihoods = log_y_likelihoods.view(-1, nb_mc_integration_steps)
        log_marginal_likelihood = torch.logsumexp(log_y_likelihoods, dim=1).sum()
        return log_marginal_likelihood

    @staticmethod
    def infer(observations, prior_network, optimizer, log_likelihood_fct, source_data_dim, nb_epochs=300, batch_size=128,
              nb_mc_integration_steps=10, validation_set_size=0.1, early_stopping=True, early_stopping_patience=10,
              early_stopping_verbose=True):

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
                log_marginal_likelihood = McBiasedEstimator.compute_estimator(batch, nb_mc_integration_steps,
                                                                              batch.device, source_data_dim,
                                                                              prior_network, log_likelihood_fct)

                optimizer.zero_grad()
                loss = - log_marginal_likelihood
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            training_loss.append(np.mean(batch_loss))

            # Validate
            prior_network.eval()

            batch_loss = []
            dataloader = create_dataloader(validation_set, batch_size)
            for batch in dataloader:
                log_marginal_likelihood = McBiasedEstimator.compute_estimator(batch, nb_mc_integration_steps,
                                                                              batch.device, source_data_dim,
                                                                              prior_network, log_likelihood_fct)
                loss = - log_marginal_likelihood
                batch_loss.append(loss.item())

            validation_loss.append(np.mean(batch_loss))

            prior_network.train()

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
