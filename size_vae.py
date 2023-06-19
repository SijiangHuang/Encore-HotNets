import os
import sys
import pandas as pd
import numpy as np
import datetime  
import argparse
import time
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchsummary import summary
from utils.dataset import *
from typing import List
from torch.nn import functional as F
from scipy.stats import entropy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

def train(encoder, decoder, dataloader, optimizer):
    epoch_loss, epoch_kld, epoch_recon, sample_num = 0, 0, 0, 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        mu, var = encoder(data)
        z = reparameterize(mu, var)
        y = decoder(z)
        recon_loss = F.l1_loss(y, data)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1), dim = 0)
        loss = recon_loss + kld_weight * kld_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(data)
        epoch_kld += kld_loss.item() * len(data)
        epoch_recon += recon_loss.item() * len(data)
        sample_num += len(data)

    epoch_loss /= sample_num
    epoch_recon /= sample_num
    epoch_kld /= sample_num
    return epoch_loss, epoch_recon, epoch_kld


def cramer_dis(x, y):
    cdf_x = np.cumsum(x)
    cdf_y = np.cumsum(y)
    return np.sum(np.abs(cdf_x - cdf_y)) / x.shape[0]

def js_dis(p, q):
    p = list(p)
    q = list(q)
    pq_max_len = max(len(p), len(q))
    p += [0.0] * (pq_max_len - len(p))
    q += [0.0] * (pq_max_len - len(q))
    assert (len(p) == len(q))
    m = np.sum([p, q], axis=0) / 2
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)


def model_test(encoder, decoder):
    size_dis = []
    mean_size_dis = []
    for i in range(pairs):
        size_data = sizedata[i]
        size_data = torch.tensor(size_data, dtype=torch.float).to(device).unsqueeze(0)
        mu, var = encoder(size_data)
        z = reparameterize(mu, var)
        new_size = decoder(z)
        new_size = new_size.cpu().detach().numpy().squeeze()
        new_size[new_size < 1e-3] = 0
        new_size /= new_size.sum()
        new_mean_size = (new_size * size_cdf).sum()
        size_dis.append(js_dis(new_size, sizedata[i]))
        mean_size_dis.append(np.abs(new_mean_size - mean_sizes[i]) / mean_sizes[i])
    return np.mean(size_dis), mean_size_dis


def get_dis(decoder, latent_dim, sizedata, pairs):
    size_dis = np.zeros((pairs, pairs))
    decoder.eval()
    mean_sizes = []
    for i in range(pairs):
        z = torch.randn((1, latent_dim)).to(device)
        size = decoder(z)
        size = size.squeeze().detach().to('cpu').numpy()
        size[size < 1e-3] = 0
        size /= size.sum()
        mean_size = (size * size_cdf).sum()
        mean_sizes.append(mean_size)
        for j in range(pairs):
            loss = cramer_dis(size, sizedata[j:j+1])
            size_dis[i][j] = loss
    return size_dis, mean_sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    help = "The name of dataset"
    parser.add_argument('-d', dest = "dataset", required = True, help = help)
    help = "The number of src-dst pairs"
    parser.add_argument('-n', dest = "pairs", required = True, help = help)
    args = parser.parse_args()
    pairs = int(args.pairs)

    if args.dataset == 'fb':
        pairdata, freqpairs, n_size, n_interval = get_fb_data(pairs)
        size_cdf = pd.read_csv('data/fb/size_cdf.csv')
        kld_weight = 5e-5
        lr = 1e-3

    elif args.dataset == 'univ':
        pairdata, freqpairs, n_size, n_interval = get_univ_data(pairs)
        size_cdf = pd.read_csv('data/univ/size_cdf.csv')
        kld_weight = 5e-5
        lr = 1e-3

    sizedata = get_data(pairdata, freqpairs, 'size_index', n_size)
    size_cdf = np.concatenate(([0], (size_cdf['size'].values[1:] + size_cdf['size'].values[:-1]) / 2))
    mean_sizes = (sizedata * size_cdf).sum(axis=1)
    
    # latent_dim = 32
    # hidden_dims = [64, 128, 256, 128, 64]
    latent_dim = 16
    hidden_dims = [28, 24, 20]
    encoder = SizeEncoder(n_size, hidden_dims, latent_dim).to(device)
    hidden_dims.reverse()
    decoder = SizeDecoder(n_size, hidden_dims, latent_dim).to(device)
    print('encoder:', summary(encoder, [[n_size]]))
    print('decoder:', summary(decoder, [[latent_dim]]))
    sys.stdout.flush()

    dataset = torch.tensor(sizedata, dtype=torch.float)
    # dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    # lr = 1e-3
    optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=lr)
    step_size = 20000
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.5)

    date = datetime.datetime.now()
    date = '%s-%s-%s-%s-%s' % (args.dataset, date.year, date.month, date.day, date.hour)
    if os.path.exists('model/{date}/'.format(date=date)):
        os.system('rm -r model/{date}/'.format(date=date))
    os.makedirs('model/{date}/'.format(date=date))

    f = open(os.path.join('model/{date}/log.txt'.format(date=date)), 'w')
    stop_loss = 1e-3
    encoder.train()
    decoder.train()
    start_time = time.time()
    min_epoch_loss = 100
    for epoch in range(100001):
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        epoch_loss, epoch_recon, epoch_kld = train(encoder, decoder, dataloader, optimizer)
        min_epoch_loss = min(epoch_loss, min_epoch_loss)
        if epoch and epoch % 100 == 0:
            size_dis, mean_size_dis = model_test(encoder, decoder)
            cur_time = time.time()
            print("epoch=%d, loss=%.2e, min_loss=%.2e, kld=%.2f, recon=%.2e, size_dis=%.3f, mean_size=%.2f, max_size=%.2f(%d), time=%.2f" % (epoch, epoch_loss, min_epoch_loss, epoch_kld, epoch_recon, size_dis, np.mean(mean_size_dis), np.max(mean_size_dis), np.argmax(mean_size_dis), cur_time - start_time))
            f.write("%d,%.2e,%.2e,%.2f,%.2e,%d\n" % (epoch, epoch_loss, min_epoch_loss, epoch_kld, epoch_recon, cur_time - start_time))
            sys.stdout.flush()
            f.flush()
            min_epoch_loss = 100

        if epoch and epoch % 1000 == 0:
            size_dis, pred_mean_sizes = get_dis(decoder, latent_dim, sizedata, pairs)
            accuracy = np.min(size_dis, axis=1)
            coverage = np.min(size_dis, axis=0)
            acc_percentile = np.percentile(accuracy, [90, 95, 99])
            cov_percentile = np.percentile(coverage, [90, 95, 99])
            print("\n%d, %d, acc=%.2e, %.2e, %.2e, cov=%.2e, %.2e, %.2e\n" % (np.mean(pred_mean_sizes), np.mean(mean_sizes), acc_percentile[0], acc_percentile[1], acc_percentile[2], cov_percentile[0], cov_percentile[1], cov_percentile[2]))
            sys.stdout.flush()
        
            torch.save(encoder, 'model/{date}/encoder-{i}.pth'.format(i=epoch, date=date))
            torch.save(decoder, 'model/{date}/decoder-{i}.pth'.format(i=epoch, date=date))
        
        if epoch_loss <= stop_loss:
            size_dis, mean_size_dis = model_test(encoder, decoder)
            print("epoch=%d, loss=%.2e, min_loss=%.2e, kld=%.2f, recon=%.2e, size_dis=%.3f, mean_size=%.2f, max_size=%.2f(%d), time=%.2f" % (epoch, epoch_loss, min_epoch_loss, epoch_kld, epoch_recon, size_dis, np.mean(mean_size_dis), np.max(mean_size_dis), np.argmax(mean_size_dis), cur_time - start_time))

            size_dis, pred_mean_sizes = get_dis(decoder, latent_dim, sizedata, pairs)
            accuracy = np.min(size_dis, axis=1)
            coverage = np.min(size_dis, axis=0)
            acc_percentile = np.percentile(accuracy, [90, 95, 99])
            cov_percentile = np.percentile(coverage, [90, 95, 99])
            print("\n%d, %d, acc=%.2e, %.2e, %.2e, cov=%.2e, %.2e, %.2e\n" % (np.mean(pred_mean_sizes), np.mean(mean_sizes), acc_percentile[0], acc_percentile[1], acc_percentile[2], cov_percentile[0], cov_percentile[1], cov_percentile[2]))

            sys.stdout.flush()
            torch.save(encoder, 'model/{date}/encoder-final.pth'.format(date=date))
            torch.save(decoder, 'model/{date}/decoder-final.pth'.format(date=date))
            break
    f.close()

