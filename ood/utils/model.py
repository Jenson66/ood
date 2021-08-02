from abc import abstractmethod

import torch.nn as nn
import torch.optim as optim
import torchvision

from ood.utils.errors import MisconfigurationError

class Model(nn.Module):
    '''Abstract model that is compatible with this package'''
    def __init__(self, lr, device):
        '''
        Parameters
        ----------
        lr: float
            Learning rate to use
        device: str, torch.device
            Device to run ML operations on
        '''
        super().__init__()
        self.lr = lr
        self.device = device
        self.epoch_count = 0
        self.optimizer = None

    def init_optimizer(self):
        self.optimizer = optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=0.0001
        )

    @abstractmethod
    def forward(self, *x):
        """The torch prediction function.

        Parameters
        ----------
        x: Input vector
        """
        pass

    def fit(self, data, epochs=1, scaling=1, verbose=True):
        """
        Fit the model for some epochs, return history of loss values and the
        gradients of the changed parameters

        Parameters
        ----------
        data: Iterable
            x, y pairs
        epochs: int
            number of epochs to train for
        verbose: bool
            output training stats if True
        """
        criterion = nn.CrossEntropyLoss()
        data_count = 0
        for i in range(epochs):
            self.optimizer.zero_grad()
            x, y = next(iter(data))
            x = x.to(self.device)
            y = y.to(self.device)
            output = self(x)
            loss = criterion(output, y)
            if verbose:
                print(
                    f"Epoch {i + 1}/{epochs} loss: {loss}",
                    end="\r"
                )
            loss.backward()
            self.optimizer.step()
            data_count += len(y)
        self.epoch_count += 1
        if verbose:
            print()
        return (
            data_count,
            [scaling * -self.lr * p.grad for p in self.parameters()]
        )

    def get_params(self):
        """Get the tensor form parameters of this model."""
        return [p.data for p in self.parameters()]

    def copy_params(self, params):
        """Copy input parameters into self.

        Parameters
        ----------
        params: Global model parameters list
        """
        for p, t in zip(params, self.parameters()):
            t.data.copy_(p)


class SoftMaxModel(Model):
    """The softmax perceptron class"""
    def __init__(self, lr, device, params):
        super().__init__(lr, device)
        self.features = nn.ModuleList([
            nn.Linear(
                params['num_in'], params['num_in'] * params['params_mul']
            ),
            nn.Sigmoid(),
            nn.Linear(
                params['num_in'] * params['params_mul'], params['num_out']
            ),
            nn.Softmax(dim=1)
        ]).eval()
        super().init_optimizer()

    def forward(self, x):
        for feature in self.features:
            x = feature(x)
        return x


class SqueezeNet(Model):
    """The SqueezeNet DNN Class"""
    def __init__(self, lr, device, params):
        super().__init__(lr, device)
        net = torchvision.models.__dict__["squeezenet1_1"](pretrained=True)
        net.classifier[1] = nn.Conv2d(
            512, params['num_out'], kernel_size=(1, 1), stride=(1, 1)
        )
        self.features = nn.ModuleList(
            [f for f in net.features] +
            [f for f in net.classifier]
        ).eval()
        super().copy_params([p.data for p in net.parameters()])
        super().init_optimizer()

    def forward(self, x):
        for feature in self.features:
            x = feature(x)
        return x.flatten(1)


def load_model(model_name, lr, device, arch_params):
    """Load the model as specified by the arguments

    Parameters
    ----------
    model_name: str
        Name of the model to load
    lr: float
        Learning rate to use
    device: str, torch.device
        Device to load onto
    arch_params: dict
        Extra parameters subject to the model to load
    """
    models = {
        "softmax": SoftMaxModel,
        "squeezenet": SqueezeNet,
    }
    if (chosen_model := models.get(model_name)) is None:
        raise MisconfigurationError(
            f"Model '{model_name}' does not exist, " +
            f"possible options: {set(models.keys())}"
        )
    return chosen_model(lr, device, arch_params).to(device)
