import torch
from dataclasses import dataclass
from enum import Enum, auto
from argparse import ArgumentParser, Namespace

class ModelType(Enum):
    RNN = auto()
    LSTM = auto()
    GRU = auto()
    DelayedRNN = auto()
    
def get_args():
    parser = ArgumentParser(description="Delayed RNN for QSWAP Task")
    parser.add_argument("--model_type", type=str, default="DelayedRNN", choices=["RNN", "LSTM", "GRU", "DelayedRNN"], help="Type of model to use")
    parser.add_argument("--max_delay", type=int, default=10, help="Maximum delay for DelayedRNN")
    parser.add_argument("--max_think_steps", type=int, default=100, help="Maximum thinking steps for the model")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--input_size", type=int, default=1, help="Input size (e.g., pixel value)")
    parser.add_argument("--seq_length", type=int, default=784, help="Sequence length (e.g., 28x28 for MNIST)")
    parser.add_argument("--seq_min", type=int, default=5, help="Minimum sequence length for training data")
    parser.add_argument("--seq_max", type=int, default=20, help="Maximum sequence length for training data")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of the model")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of output classes")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    
    args = parser.parse_args()
    args.model_type = ModelType[args.model_type]  # Convert string to ModelType enum
    
    return args

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
    
    @staticmethod
    def from_args(args: Namespace) -> 'Config':
        return Config(**vars(args))