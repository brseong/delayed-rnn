import torch
from dataclasses import dataclass
from enum import Enum, auto

# class Singleton(type):
#     _instances = {}
#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
#         return cls._instances[cls]

class ModelType(Enum):
    RNN = auto()
    LSTM = auto()
    GRU = auto()
    Mamba = auto()
    Transformer = auto()
    DelayedRNN = auto()

@dataclass()
class Config():
    model_type: ModelType = ModelType.DelayedRNN
    max_delay: int = 10
    max_think_steps: int = 100
    seed: int|None = None
    batch_size: int = 64
    input_size: int = 1         # 픽셀 하나씩 입력 (시퀀스 데이터)
    seq_length: int = 784       # 28x28 = 784
    seq_min: int = 5
    seq_max: int = 20
    hidden_size: int = 128
    num_classes: int = 10
    learning_rate: float = 0.001
    epochs: int = 10
    device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')