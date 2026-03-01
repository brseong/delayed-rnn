import os
import torch
import random
import numpy as np

TRAIN_LOSS = "train/loss"
TRAIN_TOKEN_ACC = "train/token_accuracy"
TRAIN_SEQ_ACC = "train/sequence_accuracy"
TRAIN_TIME = "train/time_per_data"
TRAIN_GRAD_NORM = "train/grad_norm"

EVAL_LOSS = "eval/loss"
EVAL_TOKEN_ACC = "eval/token_accuracy"
EVAL_SEQ_ACC = "eval/sequence_accuracy"
EVAL_FLOPS_1B = "eval/Flops(1B)"
EVAL_TIME = "eval/time_per_data"


def set_seed(seed: int):
    random.seed(seed) # Python 내장 random
    np.random.seed(seed) # Numpy
    torch.manual_seed(seed) # PyTorch CPU
    torch.cuda.manual_seed(seed) # PyTorch GPU
    torch.cuda.manual_seed_all(seed) # Multi-GPU
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)