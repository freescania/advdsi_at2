import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class PytorchMultiClass(nn.Module):
    def __init__(self, num_features):
        super(PytorchMultiClass, self).__init__()

        self.layer_1 = nn.Linear(num_features, 32)
        self.layer_out = nn.Linear(32, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
        x = self.layer_out(x)
        return self.softmax(x)


def train_classification(train_data, model, criterion, optimizer, batch_size, device, scheduler=None,
                         generate_batch=None):
    """Train a Pytorch multi-class classification model

    Parameters
    ----------
    train_data : torch.utils.data.Dataset
        Pytorch dataset
    model: torch.nn.Module
        Pytorch Model
    criterion: function
        Loss function
    optimizer: torch.optim
        Optimizer
    batch_size : int
        Number of observations per batch
    device : str
        Name of the device used for the model
    scheduler : torch.optim.lr_scheduler
        Pytorch Scheduler used for updating learning rate
    collate_fn : function
        Function defining required pre-processing steps

    Returns
    -------
    Float
        Loss score
    Float:
        Accuracy Score
    """

    # Set model to training mode
    model.train()
    train_loss = 0
    train_acc = 0

    # Create data loader
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)

    # Iterate through data by batch of observations
    for feature, target_class in data:
        # Reset gradients
        optimizer.zero_grad()

        # Load data to specified device
        feature, target_class = feature.to(device), target_class.to(device)

        # Make predictions
        output = model(feature)

        # Calculate loss for given batch
        loss = criterion(output, target_class.long())

        # Calculate global loss
        train_loss += loss.item()

        # Calculate gradients
        loss.backward()

        # Update Weights
        optimizer.step()

        # Calculate global accuracy
        train_acc += (output.argmax(1) == target_class).sum().item()

    # Adjust the learning rate
    if scheduler:
        scheduler.step()

    return train_loss / len(train_data), train_acc / len(train_data)


def test_classification(test_data, model, criterion, batch_size, device, generate_batch=None):
    """Calculate performance of a Pytorch multi-class classification model

    Parameters
    ----------
    test_data : torch.utils.data.Dataset
        Pytorch dataset
    model: torch.nn.Module
        Pytorch Model
    criterion: function
        Loss function
    bacth_size : int
        Number of observations per batch
    device : str
        Name of the device used for the model
    collate_fn : function
        Function defining required pre-processing steps

    Returns
    -------
    Float
        Loss score
    Float:
        Accuracy Score
    """

    # Set model to evaluation mode
    model.eval()
    test_loss = 0
    test_acc = 0

    # Create data loader
    data = DataLoader(test_data, batch_size=batch_size, collate_fn=generate_batch)

    # Iterate through data by batch of observations
    for feature, target_class in data:
        # Load data to specified device
        feature, target_class = feature.to(device), target_class.to(device)

        # Set no update to gradients
        with torch.no_grad():
            # Make predictions
            output = model(feature)

            # Calculate loss for given batch
            loss = criterion(output, target_class.long())

            # Calculate global loss
            test_loss += loss.item()

            # Calculate global accuracy
            test_acc += (output.argmax(1) == target_class).sum().item()

    return test_loss / len(test_data), test_acc / len(test_data)