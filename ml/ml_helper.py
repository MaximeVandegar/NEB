from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch


def fit_conditional_normalizing_flow(network, optimizer, y, x, early_stopping_verbose=True, validation_size=0.2,
                                     early_stopping_patience=10, batch_size=256, nb_epochs=300, random_state=0):
    x_dim = x.shape[1]
    y_train, y_test, x_train, x_test = train_test_split(y, x, test_size=validation_size, random_state=random_state)
    training_dataset = DataLoader(torch.cat((x_train, y_train), dim=1), batch_size=batch_size, shuffle=True)
    validation_dataset = DataLoader(torch.cat((x_test, y_test), dim=1), batch_size=batch_size, shuffle=True)

    training_loss = []
    validation_loss = []
    early_stopping_count = 0
    best_validation_loss = float('inf')

    for epoch in tqdm(range(nb_epochs)):

        # Training
        batch_losses = []
        for batch in training_dataset:
            x = batch[:, :x_dim]
            y = batch[:, x_dim:]
            ll, z = network.compute_ll(y, context=x)

            optimizer.zero_grad()
            loss = - torch.mean(ll)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
        training_loss.append(np.mean(batch_losses))

        # Validation
        with torch.no_grad():

            network.eval()

            batch_losses = []
            for batch in validation_dataset:
                x = batch[:, :x_dim]
                y = batch[:, x_dim:]
                ll, z = network.compute_ll(y, context=x)
                loss = - torch.mean(ll)

                batch_losses.append(loss.item())
            validation_loss.append(np.mean(batch_losses))

            network.train()

        if validation_loss[-1] < best_validation_loss:
            early_stopping_count = 0
            best_validation_loss = validation_loss[-1]
        else:
            early_stopping_count += 1
            if early_stopping_count > early_stopping_patience:
                if early_stopping_verbose:
                    print(
                        f'Validation loss did not improve for {early_stopping_patience} epochs, '
                        f'stopping training after {epoch} epochs.')
                break

    return training_loss, validation_loss
