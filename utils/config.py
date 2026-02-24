import torch
from dataclasses import dataclass
from enum import Enum, auto

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class ModelType(Enum):
    RNN = auto()
    LSTM = auto()
    LEARNABLE_DELAY_RNN = auto()

@dataclass
class Config(metaclass=Singleton):
    model_type = ModelType.LEARNABLE_DELAY_RNN
    max_delay = 10
    
    seed = 42
    batch_size = 64
    input_size = 1         # 픽셀 하나씩 입력 (시퀀스 데이터)
    seq_length = 784       # 28x28 = 784
    seq_min = 5
    seq_max = 20
    hidden_size = 128
    num_classes = 10
    learning_rate = 0.001
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')