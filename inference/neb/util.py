import torch


def create_dataloader(dataset, batch_size):
    """
    Split the dataset into stochastic minibatches
    """
    r = torch.randperm(dataset.shape[0])
    dataset = dataset[r]
    dataloader = torch.split(dataset, batch_size)
    return dataloader
