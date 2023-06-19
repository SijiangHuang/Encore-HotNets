import os
import sys
import pandas as pd
import numpy as np
import datetime  
import time
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchsummary import summary
from utils.dataset import *
from typing import List
from torch.nn import functional as F
from scipy.stats import entropy
from collections import defaultdict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SizeToHidden(nn.Module):
    def __init__(self, input_size, hidden_dims, hidden_size, n_layer):
        super(SizeToHidden, self).__init__()
        self.lins = nn.ModuleList()
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        in_dim = input_size 
        for h_dim in hidden_dims:
            self.lins.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_features=h_dim),
                    nn.ReLU())
            )
            in_dim = h_dim
        self.output = nn.Linear(in_dim, out_features=hidden_size * n_layer)

    def forward(self, x: Tensor) -> List[Tensor]:
        for lin in self.lins:
            x = lin(x)
        x = self.output(x)
        x = x.view(-1, self.n_layer, self.hidden_size)
        x = x.permute(1, 0, 2).contiguous()
        return x


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, n_layer)
        self.h2o = nn.Linear(hidden_size, n_size)
        self.softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
        out = self.h2o(out)
        out = self.softmax(out)
        return out, hidden

def sample_dataset(seed):
    np.random.seed(seed)
    dataset = []
    for pair in range(pairs):
        ran_index = np.random.randint(len(seq_set[pair]))
        dataset.append([seq_set[pair][ran_index], size_set[pair], target_set[pair][ran_index]])
    return dataset

def inputTensor(lines):
    tensor = torch.zeros(lines.shape[1], lines.shape[0], n_size, dtype=torch.long)
    for line in range(lines.shape[0]):
        for i in range(lines.shape[1]):
            size = lines[line][i]
            tensor[i][line][size] = 1
    return tensor

def train(s2h, gru, dataloader, optimizer):
    gru.train()
    s2h.train()
    sum_loss = 0
    for seq_tensor, size_tensor, target_tensor in dataloader:
        seq_tensor = inputTensor(seq_tensor).float().to(device)
        size_tensor = size_tensor.float().to(device)
        target_tensor = target_tensor.T.long().to(device)
        optimizer.zero_grad()
        output, hn = gru(seq_tensor, s2h(size_tensor))
        loss = 0
        for i in range(seq_len):
            loss += nn.NLLLoss()(output[i], target_tensor[i])
        loss.backward()
        optimizer.step()
        sum_loss += loss.item() / seq_tensor.shape[0] * seq_tensor.shape[1]
    return sum_loss / len(dataloader.dataset)

if __name__ == "__main__":
    # pairs = 1000
    # seq_len = 16
    # pairdata, freqpairs, n_size, n_interval = get_univ_data(pairs)
    # sizedata = get_data(pairdata, freqpairs, 'size_index', n_size)

    pairs = 1000
    seq_len = 16
    pairdata, freqpairs, n_size, n_interval = get_univ_data(pairs)
    sizedata = get_data(pairdata, freqpairs, 'size_index', n_size)

    seq_set = defaultdict(list)
    target_set = defaultdict(list)
    size_set = {}

    for pair in range(pairs):
        size_index = pairdata[freqpairs[pair]].size_index.values
        target_index = np.concatenate((size_index[1:], size_index[0:1]))
        for i in range(len(size_index) - seq_len):
            seq_set[pair].append(size_index[i:i+seq_len])
            target_set[pair].append(target_index[i:i+seq_len])
            size_set[pair] = sizedata[pair]
        seq_set[pair] = np.array(seq_set[pair])
        target_set[pair] = np.array(target_set[pair])
    
    hidden_size = 512
    gru = GRU(n_size, hidden_size, 1).to(device)
    s2h = SizeToHidden(n_size, [128, 256], hidden_size, 1).to(device)

    lr = 1e-3
    optimizer = torch.optim.Adam([{'params': gru.parameters()}, {'params': s2h.parameters()}], lr=lr)
    step_size = 20000
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.5)

    date = datetime.datetime.now()
    date = 'univ-gru-%s-%s-%s-%s' % (date.year, date.month, date.day, date.hour)
    if os.path.exists('model/{date}/'.format(date=date)):
        os.system('rm -r model/{date}/'.format(date=date))
    os.makedirs('model/{date}/'.format(date=date))

    stop_loss = 0.2
    s_time = time.time()
    plot_every = 100
    save_every = 1000
    avg_loss = 0
    for i in range(100001):
        dataset = sample_dataset(i)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        loss = train(s2h, gru, dataloader, optimizer)
        avg_loss += loss

        if i and i % plot_every == 0:
            print(i, loss, avg_loss / plot_every, time.time() - s_time)
            sys.stdout.flush()
            if avg_loss / plot_every < stop_loss:
                print(i, avg_loss / plot_every)
                torch.save(s2h, 'model/{date}/s2h-final.pth'.format(date=date))
                torch.save(gru, 'model/{date}/gru-final.pth'.format(date=date))
                break
            avg_loss = 0

        if i and i % save_every == 0:
            torch.save(s2h, 'model/{date}/s2h-{i}.pth'.format(i=i, date=date))
            torch.save(gru, 'model/{date}/gru-{i}.pth'.format(i=i, date=date))