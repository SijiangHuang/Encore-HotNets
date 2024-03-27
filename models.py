import torch
from typing import List
from torch import nn, Tensor
from torch.nn import functional as F


class SizeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(SizeEncoder, self).__init__()
        self.encoder = nn.ModuleList()
        in_dim = input_dim 
        for h_dim in hidden_dims:
            self.encoder.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_features=h_dim),
                    nn.ReLU())
            )
            in_dim = h_dim
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, x: Tensor) -> List[Tensor]:
        for module in self.encoder:
            x = module(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return [mu, log_var]


class SizeDecoder(torch.nn.Module):
    def __init__(self, output_dim, hidden_dims, latent_dim):
        super(SizeDecoder, self).__init__()
        self.decoder = torch.nn.ModuleList()
        in_dim = latent_dim
        for h_dim in hidden_dims:
            self.decoder.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_features=h_dim,),
                    nn.ReLU())
            )
            in_dim = h_dim
        self.output = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, x: Tensor) -> List[Tensor]:
        for module in self.decoder:
            x = module(x)
        result = self.output(x)
        result = F.softmax(result, dim=1)
        return result


class SizeToHidden(nn.Module):
    def __init__(self, n_layer, hidden_size, input_size, hidden_dims):
        """
        Initialize SizeToHidden module.

        Args:
            n_layer (int): Number of layers.
            hidden_size (int): Size of hidden layers.
            input_size (int): Size of input.
            hidden_dims (list): List of sizes of hidden layers.
        """
        super(SizeToHidden, self).__init__()
        self.n_layer = n_layer
        layers = []
        in_dim = input_size
        for h_dim in hidden_dims:
            layers.append(nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ))
            in_dim = h_dim
        self.lins = nn.ModuleList(layers)
        self.output = nn.Linear(in_dim, hidden_size)

    def forward(self, x):
        """
        Forward pass SizeToHidden module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        for lin in self.lins:
            x = lin(x)
        x = self.output(x)
        x = x.unsqueeze(0).repeat(self.n_layer, 1, 1)
        return x


class GRU(nn.Module):
    def __init__(self, hidden_size, n_layer, embed_size, input_size):
        """
        Initialize GRU module.

        Args:
            hidden_size (int): Size of hidden layers.
            n_layer (int): Number of layers.
            embed_size (int): Size of embedding.
            input_size (int): Size of input.
        """
        super(GRU, self).__init__()
        self.gru = nn.GRU(embed_size, hidden_size, n_layer)
        self.h2o = nn.Linear(hidden_size, input_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.embedding = nn.Embedding(input_size, embed_size)

    def forward(self, x, hidden):
        """
        Forward pass GRU module.

        Args:
            x (Tensor): Input tensor.
            hidden (Tensor): Hidden state tensor.

        Returns:
            Tensor: Output tensor.
            Tensor: Hidden state tensor.
        """
        x = self.embedding(x).permute(1, 0, 2)
        out, hidden = self.gru(x, hidden)
        out = self.h2o(F.relu(out))
        out = self.softmax(out)
        return out, hidden


class Model(nn.Module):
    def __init__(self, gru_params, s2h_params):
        """
        Initialize Model module.

        Args:
            gru_params (dict): Parameters for GRU module.
            s2h_params (dict): Parameters for SizeToHidden module.
        """
        super(Model, self).__init__()
        self.gru = GRU(**gru_params)
        self.s2h = SizeToHidden(**s2h_params)

    def forward(self, x):
        """
        Forward pass Model module.

        Args:
            x (tuple): Input tuple containing seq_tensor and size_interval.

        Returns:
            Tensor: Output tensor.
        """
        seq_tensor, size_interval = x
        hidden = self.s2h(size_interval)
        out, hidden = self.gru(seq_tensor, hidden)
        return out


