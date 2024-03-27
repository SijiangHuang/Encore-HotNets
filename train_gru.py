import os
import sys
import time 
import datetime
import numpy as np
import pandas as pd
from models import *
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.spatial.distance import jensenshannon
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_trainset(data, block_size, n_size, min_samples, device):
    """
    Generate training set.

    Args:
        data (list): List of sequences.
        block_size (int): Size of sequence blocks.
        n_size (int): Number of size intervals.
        min_samples (int): Minimum number of samples.
        device (torch.device): Device for data placement.
    Returns:
        list: List of training samples.
    """
    trainset = []
    for seq in data:
        seq_trainset = []
        seq = np.append(seq, seq[0:block_size-1])
        size_data = compute_probability_distribution(seq, n_size)
        num_samples = max(min_samples, len(seq) - block_size)
        for i in range(num_samples):
            index = i % (len(seq) - block_size)
            sequence_block = seq[index:index+block_size]
            target_sequence = seq[index+1:index+block_size+1]
            seq_trainset.append([
                torch.from_numpy(sequence_block).to(device, non_blocking=True).long(),
                torch.from_numpy(target_sequence).to(device, non_blocking=True).long(),
                torch.from_numpy(size_data).to(device, non_blocking=True).float(),
            ])
        trainset.extend(seq_trainset)
    return trainset


def train_batch(model, dloader, optimizer):
    """
    Train the model for one batch.

    Args:
        model (torch.nn.Module): The model to be trained.
        dloader (DataLoader): DataLoader providing batches of data.
        optimizer (torch.optim.Optimizer): Optimizer used for parameter updates.

    Returns:
        float: The loss value for the current batch.
    """
    model.train()  # Ensure the model is in training mode.
    seq_tensor, target_tensor, size_tensor = next(iter(dloader))  # Extract batch data.
    target_tensor = target_tensor.T  # Transpose the target tensor to match output dimensions.
    
    optimizer.zero_grad()  # Clear existing gradients.
    output = model((seq_tensor, size_tensor))  # Forward pass.
    
    # Compute loss
    loss = F.nll_loss(output.contiguous().view(-1, output.size(-1)), 
                      target_tensor.contiguous().view(-1), 
                      reduction='mean')
    loss.backward()  # Backward pass.
    optimizer.step()  # Update model parameters.
    
    return loss.item()  # Return the loss value.


def get_data_loader(trainset, subset_size, batch_size, seed):
    """
    Creates a DataLoader for training with a subset of the full dataset.

    Args:
        trainset (list): Complete list of training samples.
        subset_size (int): Size of the subset to be used for training.
        batch_size (int): Batch size for training.
        seed (int): Random seed for sample shuffling and subset selection.

    Returns:
        DataLoader: DataLoader configured with the specified subset and batch size.
    """
    np.random.seed(seed)  # Set the random seed for reproducibility.
    subset_size = min(subset_size, len(trainset))  # Ensure subset size does not exceed trainset size.
    
    # Randomly select indices for the subset.
    ran_index = np.random.choice(len(trainset), subset_size, replace=False)
    subset_trainset = [trainset[i] for i in ran_index]  # Create subset based on selected indices.
    
    return DataLoader(subset_trainset, batch_size=batch_size, shuffle=True)  # Return DataLoader for the subset.


def train_model(model, lr, plot_every, max_epoch, subset_size=100000, stop_loss=0.2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.9)
    s_time = time.time()
    avg_loss = 0
    train_loader = get_data_loader(trainset, subset_size, batch_size, seed=0)
    for epoch in range(max_epoch):
        loss = train_batch(model, train_loader, optimizer)
        avg_loss += loss
        if epoch and epoch % plot_every == 0:
            print('epoch: {}, loss: {:.4f}, lr={:.2e}, time: {:3d}min {:02d}sec'.format(epoch, avg_loss / plot_every, optimizer.param_groups[0]['lr'], int(time.time() - s_time) // 60, int(time.time() - s_time) % 60))
            sys.stdout.flush()
            if avg_loss / plot_every < stop_loss:
                print('early stopping with loss {:.4f}'.format(avg_loss / plot_every))
                break
            avg_loss = 0
            torch.cuda.empty_cache()
            train_loader = get_data_loader(trainset, subset_size, batch_size, seed=epoch)
        if optimizer.param_groups[0]['lr'] > 1e-4:
            scheduler.step()



if __name__ == "__main__":
    print('start training in {}-{}-{} {}:{}:{:02d}'.format(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day, datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second))
    sys.stdout.flush()

    # read data
    size_file = '/home/Encore-HotNets/data/univ/flowsize.txt'
    size_cdf = pd.read_csv('/home/Encore-HotNets/data/univ/size_cdf.csv')
    n_size = len(size_cdf) - 1
    block_size = 16
    batch_size = 64
    min_samples = 200
    data = get_data(size_file)

    # make model directory
    date = datetime.datetime.now()
    model_path = 'univ-gru-%s-%s-%s-%s' % (date.year, date.month, date.day, date.hour)
    trainset = get_trainset(data, block_size, n_size, min_samples, device)
    print('Training model with trainset of size {}'.format(len(trainset)))

    gru_params = {
        'hidden_size': 512,
        'n_layer': 1,
        'embed_size': 128,
        'input_size': n_size,
    }
    s2h_params = {
        'n_layer': 1,
        'hidden_size': 512,
        'input_size': n_size,
        'hidden_dims': [128, 256]
    }
    model = Model(gru_params, s2h_params).to(device)
    train_model(model, lr=1e-3, plot_every=10000, max_epoch=1000001, subset_size=50000, stop_loss=0.2)
    ckpt = {
        'model': model.state_dict(),
        'gru_params': gru_params,
        's2h_params': s2h_params,
    }
    torch.save(ckpt, 'model/{path}.pt'.format(path=model_path))
    print('Training finished in {}-{}-{} {}:{}:{:02d}'.format(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day, datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second))

            