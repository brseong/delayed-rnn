import time
import torch
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

TF32 = torch.float32
TL = torch.long
Tensor = torch.Tensor

from model import compute_loss


class Qswap(Dataset):
    def __init__(
        self, 
        train_size: int, 
        eval_size: int, 
        one_hot_dim: int, 
        max_seq_len: int, 
        min_seq_len: int, 
        job_type: str,
        train: bool
    )-> None:
        """_summary_
        Args:
            train_size (int): Number of training samples in the dataset.
            eval_size (int): Number of evaluation samples in the dataset.
            one_hot_dim (int): Dimensionality of one-hot encoded vectors.
            max_seq_len (int): Maximum sequence length in the dataset.
            min_seq_len (int): Minimum sequence length in the dataset.
            job_type (str): Type of job (e.g., "non_streaming_seq2seq", "streaming_seq2seq", "classification").
            train (bool): Whether this is a training dataset or not.
        """
        super().__init__()
        self.size: int = train_size if train else eval_size
        self.max_seq_len: int = max_seq_len
        self.min_seq_len: int = min_seq_len
        self.num_classes: int = one_hot_dim
        self.num_sep_tokens: int = 3 # U can change this if you want to add more command tokens
        self.real_min_seq_len: int = self.min_seq_len + self.num_sep_tokens
        self.real_max_seq_len: int = self.max_seq_len + self.num_sep_tokens
        
        # Dataset-specific attributes
        self.dataset_name: str = self.__class__.__name__
        self.input_size: int = one_hot_dim + 1
        self.output_size: int = one_hot_dim 
        self.job_type: str = job_type
        
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx):
        seq_len = random.randint(self.real_min_seq_len, self.real_max_seq_len)
        seq_len_wo_sep = seq_len - self.num_sep_tokens
        seq_indices = [random.randint(0, self.input_size - 2) for _ in range(seq_len_wo_sep)]
        pos1, pos2 = random.sample(range(seq_len_wo_sep), 2)
        
        # Create input vectors with one-hot encoding
        input_vectors: list[list[float]] = []
        for s_idx in seq_indices:
            vec: list[float] = [0.0] * self.input_size
            vec[s_idx] = 1.0 
            input_vectors.append(vec)
        
        # Append SEP tokens 
        sep_vec: list[float] = [0.0] * self.input_size
        sep_vec[self.input_size - 1] = 1.0
        input_vectors.append(sep_vec)
        
        cmd1_vec = self.int_to_binary_vec(pos1) + [0.0]
        input_vectors.append(cmd1_vec)
        cmd2_vec = self.int_to_binary_vec(pos2) + [0.0]
        input_vectors.append(cmd2_vec)
        
        if len(input_vectors) < self.real_max_seq_len:
            padding_len = self.real_max_seq_len - len(input_vectors)
            for _ in range(padding_len):
                input_vectors.append([0.0] * self.input_size)
                
        assert len(input_vectors) == self.real_max_seq_len, f"input_vectors length {len(input_vectors)} != real_max_seq_len {self.real_max_seq_len}"
        
        target_seq: list[int] = seq_indices[:]
        target_seq[pos1], target_seq[pos2] = target_seq[pos2], target_seq[pos1]

        if len(target_seq) < self.max_seq_len:
            for _ in range(self.max_seq_len - len(target_seq)):
                target_seq.append(0)
                
        assert len(target_seq) == self.max_seq_len, f"target_seq length {len(target_seq)} != max_seq_len {self.max_seq_len}"
        
        input_length: int = int(seq_len)
        output_length: int = int(seq_len - self.num_sep_tokens)

        return (
            torch.tensor(input_vectors, dtype=TF32), 
            torch.tensor(target_seq, dtype=TL), 
            input_length,
            output_length
        )
        
    def get_collate_fn(self) -> callable:
        def collate_fn(batch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            inputs, targets, lengths, output_lengths = zip(*batch)
            padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
            padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
            lengths = torch.tensor(lengths, dtype=TL)
            output_lengths = torch.tensor(output_lengths, dtype=TL)
            return padded_inputs, padded_targets, lengths, output_lengths
        return collate_fn
    
    def int_to_binary_vec(self, n):
        binary_str: str = format(n, f'0{self.input_size - 1}b') # Convert integer to binary string with leading zeros
        if len(binary_str) > self.input_size - 1: 
            raise ValueError(f"Integer {n} cannot be represented in {self.input_size - 1} bits.")
        return [float(b) for b in binary_str]

    def fill_logs(self, model_log_dict: dict, logs: dict):
        if logs is None:
            return model_log_dict
        for log_key, log_value in logs.items():
            if log_key not in model_log_dict:
                model_log_dict[log_key] = []
            if isinstance(log_value, torch.Tensor):
                model_log_dict[log_key].append(log_value.item())
            elif isinstance(log_value, (float, int)):
                model_log_dict[log_key].append(log_value)
            else:
                raise ValueError(f"Unsupported log value type: {type(log_value)} for key: {log_key}")
        return model_log_dict
    
    def model_log2log(self, model_log_dict: dict, sec: str):
        log_dict = {}
        for key, values in model_log_dict.items():
            if isinstance(values[0], (float, int)):
                if "/" in key: # If the key already contains a section, we don't add another section prefix
                    log_dict[key] = sum(values) / len(values)
                else:
                    log_dict[f"{sec}/{key}"] = sum(values) / len(values)
            else:
                raise ValueError(f"Unsupported log value type in model_log_dict for key: {key}")
        return log_dict
    
    def train(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
        device: str
    ):
        model.train()
        
        model_log_dict: dict[str, list[float]] = {}
        
        for _, batch in enumerate(dataloader):
            inputs: Tensor[TF32] = batch[0].to(device)
            targets: Tensor[TL] = batch[1].to(device)
            lengths: Tensor[TL] = batch[2].to(device)
            out_lengths: Tensor[TL] = batch[3].to(device)
            
            start_time = time.time()
            out, logs = model(
                inputs, 
                lengths=lengths, 
                out_lengths=out_lengths.max().item(), 
                train=True
            )
            logs = compute_loss(
                out = out, 
                model = model,
                out_lengths=out_lengths, 
                targets = targets, 
                optimizer = optimizer,
                job_type = self.job_type,
                dataset_name = self.dataset_name,
                logs = logs
            )
            logs["train_time"] = time.time() - start_time

            model_log_dict = self.fill_logs(model_log_dict=model_log_dict, logs=logs)
        
        return self.model_log2log(model_log_dict=model_log_dict, sec="train")
    
    def eval(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str
    ):
        model.eval()
        
        model_log_dict: dict[str, list[float]] = {}
        
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                inputs: Tensor[TF32] = batch[0].to(device)
                targets: Tensor[TL] = batch[1].to(device)
                lengths: Tensor[TL] = batch[2].to(device)
                out_lengths: Tensor[TL] = batch[3].to(device)

                start_time = time.time()
                out, logs = model(
                    inputs, 
                    lengths=lengths, 
                    out_lengths=out_lengths.max().item(), 
                    train=False
                )
                logs["eval_time"] = time.time() - start_time
                
                logs = compute_loss(
                    out = out, 
                    model = model,
                    out_lengths=out_lengths, 
                    targets = targets, 
                    optimizer = None,
                    job_type = self.job_type,
                    dataset_name = self.dataset_name,
                    logs = logs
                )
                
                model_log_dict = self.fill_logs(model_log_dict=model_log_dict, logs=logs)

        return self.model_log2log(model_log_dict=model_log_dict, sec="eval")