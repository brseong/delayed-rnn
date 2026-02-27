import os
import torch
import random
import numpy as np

from fvcore.nn import FlopCountAnalysis

def set_seed(seed: int):
    random.seed(seed) # Python 내장 random
    np.random.seed(seed) # Numpy
    torch.manual_seed(seed) # PyTorch CPU
    torch.cuda.manual_seed(seed) # PyTorch GPU
    torch.cuda.manual_seed_all(seed) # Multi-GPU
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
