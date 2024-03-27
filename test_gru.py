import os
import sys
import time 
import datetime
import numpy as np
import pandas as pd
from models import *
from utils import *
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_next_size(size_data: np.ndarray, seq: np.ndarray, model: nn.Module, device: torch.device, n_size: int) -> int:
    """
    Sample the next size from the distribution predicted by the model.

    Args:
        size_data (np.ndarray): The size data used as part of the model input.
        seq (np.ndarray): The current sequence of sizes used as part of the model input.
        model (nn.Module): The trained model used for prediction.
        device (torch.device): The device on which to perform the computation.
        n_size (int): The number of possible sizes to sample from.

    Returns:
        int: The next size sampled based on the model's prediction.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Convert inputs to tensors and move to the specified device
    size_tensor = torch.tensor(size_data, dtype=torch.float).unsqueeze(0).to(device)
    seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)

    # Predict the next size distribution
    with torch.no_grad():
        output = model((seq_tensor, size_tensor))
        probabilities = nn.Softmax(dim=2)(output)[-1]  # Apply softmax to the last output

    # Convert probabilities to a numpy array and sample the next size
    probabilities_np = probabilities.squeeze().cpu().numpy()
    next_size = np.random.choice(a=n_size, p=probabilities_np)

    return next_size


def generate_sequence(seq: np.ndarray, model: torch.nn.Module, block_size: int, n_size: int, device: torch.device, seq_len: int = 1000, initial_seed: int = 42) -> (np.ndarray, float):
    """
    Generate a sequence of specific length based on an initial sequence using a model.

    Args:
        seq (np.ndarray): Initial sequence to base the generation on.
        model (torch.nn.Module): Model used for generating the next size in the sequence.
        block_size (int): Size of blocks to consider for generating the next size.
        n_size (int): Number of possible sizes to sample from.
        device (torch.device): The device model and data should be on.
        seq_len (int, optional): The desired length of the generated sequence. Defaults to 1000.
        initial_seed (int, optional): Seed for random number generation for reproducibility. Defaults to 42.

    Returns:
        np.ndarray: Generated sequence of the specified length.
        float: The squared Jensen-Shannon divergence between the size distributions of the initial and generated sequences.
    """
    # Set seed for reproducibility
    np.random.seed(initial_seed)
    torch.manual_seed(initial_seed)

    # Compute the size distribution of the original sequence
    size_dist = compute_probability_distribution(seq, n_size)

    # Start the generated sequence with the first size of the original sequence
    seq_generated = [seq[0]]

    # Generate the sequence
    while len(seq_generated) < seq_len:
        current_context = seq_generated[-(block_size-1):] if len(seq_generated) >= block_size else seq_generated
        next_size = sample_next_size(size_dist, current_context, model, device, n_size)
        if size_dist[next_size] > 0:
            seq_generated.append(next_size)
        else:
            next_size = np.random.choice(a=n_size, p=size_dist)
            seq_generated.append(next_size)

    seq_generated = np.array(seq_generated[:seq_len])
    gen_dist = compute_probability_distribution(seq_generated, n_size)

    # Compute the Jensen-Shannon divergence between the original and generated distributions
    js_divergence = jensenshannon(size_dist, gen_dist, base=np.e) ** 2

    return seq_generated, js_divergence


def compute_jsds(model, sequences, block_size, input_size, device, seq_len=1000, seed=42):
    """
    Compute Jensen-Shannon Divergences (JSDs) for sequences using heuristic and model-generated sequences.

    Args:
        model (torch.nn.Module): The model to generate sequences.
        sequences (list of np.ndarray): List of original sequences.
        block_size (int): Block size used for generating sequences.
        input_size (int): Input size of the model.
        device (torch.device): The device to run the model on.
        seed (int): Seed for random number generation for reproducibility.

    Returns:
        dict: JSDs calculated using both heuristic (permuted) and model-generated sequences.
    """
    n_gram_jsds = {'heuristic': {n: [] for n in [2, 3, 4]}, 'model': {n: [] for n in [2, 3, 4]}}
    
    for seq in tqdm(sequences):
        # Prepare the sequence
        seq_padded = np.append(seq, seq[:block_size-1])
        
        # Generate permuted sequence for heuristic comparison
        seq_permuted = np.random.permutation(seq_padded)
        
        # Generate sequence using model, retry if JSD is too high
        current_seed = seed
        seq_generated, jsd = generate_sequence(seq_padded, model, block_size, input_size, device, seq_len, current_seed)
        # while jsd > 0.05:
        #     print(jsd)
        #     current_seed += 1
        #     seq_generated, jsd = generate_sequence(seq_padded, model, block_size, input_size, device, seq_len, current_seed)
        
        # Compute n-gram distributions and JSDs
        for n in [2, 3, 4]:
            n_gram_ori = compute_ngram_distribution(seq_padded, n)
            n_gram_gen = compute_ngram_distribution(seq_generated, n)
            n_gram_permuted = compute_ngram_distribution(seq_permuted, n)
            
            jsd_heuristic = compute_js_divergence(n_gram_ori, n_gram_permuted)
            jsd_model = compute_js_divergence(n_gram_ori, n_gram_gen)
            
            n_gram_jsds['heuristic'][n].append(jsd_heuristic)
            n_gram_jsds['model'][n].append(jsd_model)

    return n_gram_jsds


if __name__ == "__main__":
    print('start testing in {}-{}-{} {}:{}:{:02d}'.format(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day, datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second))
    sys.stdout.flush()

    # read data
    size_file = '/home/Encore-HotNets/data/univ/flowsize.txt'
    size_cdf = pd.read_csv('/home/Encore-HotNets/data/univ/size_cdf.csv')
    n_size = len(size_cdf) - 1
    block_size = 16
    data = get_data(size_file)

    ckpt = torch.load('model/univ-gru-2024-3-27-14.pt')
    gru_params = ckpt['gru_params']
    s2h_params = ckpt['s2h_params']
    model = Model(gru_params, s2h_params).to(device)
    model.load_state_dict(ckpt['model'])

    print('Testing model')
    n_gram_jsds = compute_jsds(model, data, block_size, n_size, device)
    result_file = 'results/jsds.txt'
    with open(result_file, 'w') as f:
        for i in range(len(data)):
            f.write('i={}, heuristic JSD: {:4f}/{:4f}/{:4f}, model JSD: {:4f}/{:4f}/{:4f}\n'.format(i, n_gram_jsds['heuristic'][2][i], n_gram_jsds['heuristic'][3][i], n_gram_jsds['heuristic'][4][i], n_gram_jsds['model'][2][i], n_gram_jsds['model'][3][i], n_gram_jsds['model'][4][i]))

    for n in [2, 3, 4]:
        print('n={}: heuristic JSD: {:.4f}, model JSD: {:.4f}'.format(n, np.mean(n_gram_jsds['heuristic'][n]), np.mean(n_gram_jsds['model'][n])))
