import os
import torch
import random
import numpy as np

from fvcore.nn import FlopCountAnalysis
from torch.utils.data import DataLoader

NS_seq2seq = "non_streaming_seq2seq"

def set_seed(seed: int)-> None:
    """ Set random seed for reproducibility across various libraries and frameworks.
    Args:
        seed (int): The random seed to set for reproducibility.
    """
    random.seed(seed) # Python 
    np.random.seed(seed) # Numpy
    torch.manual_seed(seed) # PyTorch CPU
    torch.cuda.manual_seed(seed) # PyTorch GPU
    torch.cuda.manual_seed_all(seed) # Multi-GPU
    
    torch.backends.cudnn.deterministic = True # CUDNN's deterministic mode
    torch.backends.cudnn.benchmark = False # CUDNN's benchmark mode

def calculate_flops(
    model: torch.nn.Module,
    batch_size: int,
    input_size: int,
    device: str,
    job_type: str,
):
    if job_type == NS_seq2seq:
        seq_len = 10
        input = torch.zeros(batch_size, seq_len, input_size).to(device)
        lengths = torch.tensor([seq_len] * batch_size).to(device)
        out_lengths = int(seq_len)
        train = False
        targets = None
        flops = FlopCountAnalysis(model, (input, lengths, out_lengths, train, targets))
        
    else:
        raise NotImplementedError(f"FLOPs calculation not implemented for job type: {job_type}")
        
    return flops.total() / 1e6 # Return FLOPs in millions

def init_model_compile(
    model: torch.nn.Module,
    eval_dataloader: DataLoader,
    device: str,
    job_type: str,
):
    if job_type == NS_seq2seq:
        batch = next(iter(eval_dataloader))
        input = batch[0].to(device)
        lengths = batch[2].to(device)
        out_lengths = batch[3].to(device)
        
        _, _ = model(
            x = input, 
            lengths=lengths, 
            out_lengths=out_lengths.max().item(), 
            train=False
        )

    else:
        raise NotImplementedError