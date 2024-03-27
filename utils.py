import os
import numpy as np
import pandas as pd
from collections import Counter
import torch
from torch import nn, Tensor
from scipy.spatial.distance import jensenshannon


def compute_probability_distribution(sequence, n_count):
    """
    compute probability distribution of a sequence

    Parameters:
        sequence (list): input sequence
        n_count (int): number of unique elements in the sequence

    Returns:
        np.array: probability distribution
    """
    counts = np.bincount(sequence, minlength=n_count)
    prob = counts / np.sum(counts)
    return prob


def compute_ngram_distribution(sequence, n=2):
    """
    compute n-gram distribution of a sequence
    Parameters:
        sequence (list): input sequence
        n (int): n in n-gram, default is 2

    Returns:
        dict: n-gram distribution
    """
    ngrams = [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
    ngram_counts = Counter(ngrams)
    total_count = sum(ngram_counts.values())
    ngram_dist = {k: v / total_count for k, v in ngram_counts.items()}
    return ngram_dist


def unify_distributions(dist1, dist2):
    """
    combine two distributions into one, with the same keys
    Parameters:
        dist1 (dict): first distribution
        dist2 (dict): second distribution
    Returns:
        unified_dist1 (dict): first unified distribution
        unified_dist2 (dict): second unified distribution
    """
    all_keys = set(dist1.keys()) | set(dist2.keys())
    unified_dist1 = {key: dist1.get(key, 0) for key in all_keys}
    unified_dist2 = {key: dist2.get(key, 0) for key in all_keys}
    return unified_dist1, unified_dist2
    

def compute_js_divergence(dist1, dist2):
    """
    compute Jensen-Shannon divergence between two distributions

    Parameters:
        dist1 (dict): the first distribution
        dist2 (dict): the second distribution

    Returns:
        float: Jensen-Shannon divergence
    """
    unified_dist1, unified_dist2 = unify_distributions(dist1, dist2)
    js_divergence = jensenshannon(list(unified_dist1.values()), list(unified_dist2.values()), base=np.e) ** 2
    return js_divergence


def get_data(file):
    data = []
    with open(file, 'r') as size_f:
        for size_line in size_f:
            size = np.array(list(map(int, size_line.split(','))))
            data.append(size)
    return data


def get_model_params(model):
    """
    Get total and trainable parameters of a model.
    Args:
        model (nn.Module): Model.
    Returns:
        int: Total parameters.
        int: Trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu