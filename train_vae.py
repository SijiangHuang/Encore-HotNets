import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from utils import *
from typing import List, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models import SizeEncoder, SizeDecoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(encoder: nn.Module, decoder: nn.Module, data_loader: DataLoader, optimizer: torch.optim.Optimizer, kld_weight: float) -> (float, float, float):
    """
    Train the VAE model for one epoch.

    Parameters:
    - encoder: The encoder part of the VAE.
    - decoder: The decoder part of the VAE.
    - data_loader: DataLoader providing batches of data.
    - optimizer: Optimizer used for parameter updates.
    - kld_weight: Weight for the KL divergence part of the loss.
    - device: The computing device (CPU or GPU).

    Returns:
    - epoch_loss: Average loss for the epoch.
    - epoch_recon: Average reconstruction loss for the epoch.
    - epoch_kld: Average KL divergence for the epoch.
    """
    # Set the model to training mode
    encoder.train()
    decoder.train()

    # Initialize loss accumulators
    epoch_loss = 0.0
    epoch_kld = 0.0
    epoch_recon = 0.0
    sample_num = 0

    for data in data_loader:
        # Move data to device (e.g., GPU)
        data = data.to(device)
        
        # Reset gradients to zero
        optimizer.zero_grad()

        # Forward pass through the encoder and decoder
        mu, log_var = encoder(data)
        z = reparameterize(mu, log_var)
        reconstruction = decoder(z)

        # Calculate losses
        recon_loss = F.l1_loss(reconstruction, data, reduction='mean')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / data.size(0)
        total_loss = recon_loss + kld_weight * kld_loss

        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()

        # Accumulate losses and the number of examples
        epoch_loss += total_loss.item() * data.size(0)
        epoch_kld += kld_loss.item() * data.size(0)
        epoch_recon += recon_loss.item() * data.size(0)
        sample_num += data.size(0)

    # Calculate average losses
    epoch_loss /= sample_num
    epoch_recon /= sample_num
    epoch_kld /= sample_num

    return epoch_loss, epoch_recon, epoch_kld


def test_model(decoder: torch.nn.Module, size_dists: np.ndarray, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test the VAE decoder by generating distributions and comparing them with true distributions using Cramér distance.

    Parameters:
    - decoder: The decoder part of the VAE, used for generating size distributions.
    - size_dists: Numpy array of true size distributions for comparison.
    - num_samples: Number of sample distributions to generate.
    - latent_dim: Dimensionality of the latent space.
    - device: The computing device (CPU or GPU) the decoder is running on.

    Returns:
    - acc_percentile: Percentiles of accuracy based on the minimum Cramér distance to any true distribution.
    - cov_percentile: Percentiles of coverage based on the minimum Cramér distance from any true distribution.
    """
    # Generate random latent variables
    z = torch.randn(num_samples, latent_dim).to(device)
    # Decode the latent variables to obtain size distributions
    generated_sizes = decoder(z).detach().cpu().numpy()

    # Normalize the generated distributions
    normalized_gen_sizes = [process_distribution(dist) for dist in generated_sizes]
    
    # Compute Cramér distance between generated and true distributions
    cramer_distance = cramer_dis_matrix(size_dists, np.array(normalized_gen_sizes))

    # Calculate coverage and accuracy based on Cramér distance
    coverage = np.sort(cramer_distance.min(axis=1))
    accuracy = np.sort(cramer_distance.min(axis=0))

    # Compute percentiles for accuracy and coverage
    acc_percentile = np.percentile(accuracy, [50, 75, 90, 95, 99])
    cov_percentile = np.percentile(coverage, [50, 75, 90, 95, 99])

    return acc_percentile, cov_percentile


def process_distribution(dist):
    dist[dist < 1e-3] = 0
    dist /= np.sum(dist)
    return dist


def cramer_dis_matrix(distributions_x, distributions_y):
    cdf_x = np.cumsum(distributions_x, axis=1)
    cdf_y = np.cumsum(distributions_y, axis=1)
    abs_diffs = np.abs(cdf_x[:, np.newaxis, :] - cdf_y[np.newaxis, :, :])
    cramer_distances = np.sum(abs_diffs, axis=-1) / distributions_x.shape[1]
    return cramer_distances


if __name__ == "__main__":
    print('start training in {}-{}-{} {}:{}:{:02d}'.format(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day, datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second))
    sys.stdout.flush()

    # read data
    size_file = '/home/Encore-HotNets/data/univ/flowsize.txt'
    size_cdf = pd.read_csv('/home/Encore-HotNets/data/univ/size_cdf.csv')
    n_size = len(size_cdf) - 1
    block_size = 30
    batch_size = 128
    data = get_data(size_file)

    # get size distributions
    size_dists = []
    for seq in data:
        seq_trainset = []
        seq = np.append(seq, seq[0:block_size-1])
        size_dists.append(compute_probability_distribution(seq, n_size))
    size_dists = np.array(size_dists)

    # make model directory
    date = datetime.datetime.now()
    model_path = 'univ-vae-%s-%s-%s-%s' % (date.year, date.month, date.day, date.hour)
    # if os.path.exists('model/{dir}/'.format(dir=model_dir)):
    #     os.system('rm -r model/{dir}/'.format(dir=model_dir))
    # os.makedirs('model/{dir}/'.format(dir=model_dir))

    # initialize model
    lr = 5e-3
    kld_weight = 3e-5
    latent_dim = 16
    hidden_dims = [48, 40, 32]
    encoder = SizeEncoder(n_size, hidden_dims, latent_dim).to(device)
    hidden_dims.reverse()
    decoder = SizeDecoder(n_size, hidden_dims, latent_dim).to(device)
    dataset = torch.tensor(size_dists, dtype=torch.float)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)

    # train model
    max_epoch = 50001
    test_every = 1000
    s_time = time.time()
    min_loss = 1e10
    for epoch in range(max_epoch):
        loss, recon, kld = train_model(encoder, decoder, dataloader, optimizer, kld_weight)
        scheduler.step()
        if epoch and epoch % test_every == 0:
            accuracy, coverage = test_model(decoder, size_dists, num_samples=1000)
            print('epoch={:d}, loss={:.4f}, recon={:.4f}, kld={:.4f}, time={:.2f}'.format(epoch, loss, recon, kld, time.time()-s_time))
            print('accuracy: p50={:.2e}, p75={:.2e}, p90={:.2e}, p95={:.2e}, p99={:.2e}'.format(*accuracy))
            print('coverage: p50={:.2e}, p75={:.2e}, p90={:.2e}, p95={:.2e}, p99={:.2e}'.format(*coverage))
            if loss < min_loss:
                min_loss = loss
                cpkt = {
                    'epoch': epoch,
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(cpkt, 'model/{path}.pt'.format(path=model_path))
                print('model saved at epoch %d with loss %.4f' % (epoch, loss))
            print("-"*50)    
            sys.stdout.flush()
