import torch
import random
import logging
import torch.nn.functional as F


from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from fvcore.nn import FlopCountAnalysis
logging.getLogger("fvcore.nn.jit_analysis").setLevel(logging.ERROR)


class Qswap(Dataset):
    def __init__(self, train_size = 5000, eval_size = 500, one_hot_dim = 10, min_seq_len = 10, max_seq_len = 20, train: bool = True, seed: int = 42):
        if train:
            self.size = train_size
        else:
            self.size = eval_size
            
        self.one_hot_dim = one_hot_dim
        self.min_len = min_seq_len
        self.max_len = max_seq_len
        
        self.input_size = self.one_hot_dim + 1
        self.num_classes = one_hot_dim
        self.is_classification = False
        self.dataset_name = "Qswap"

    def __len__(self):
        return self.size

    def _int_to_binary_vec(self, n):
        binary_str = format(n, f'0{self.one_hot_dim}b')
        if len(binary_str) > self.one_hot_dim: 
            binary_str = binary_str[-self.one_hot_dim:]
        return [float(b) for b in binary_str]

    def __getitem__(self, idx):
        seq_len = random.randint(self.min_len, self.max_len)
        seq_indices = [random.randint(0, self.one_hot_dim - 1) for _ in range(seq_len)]
        pos1, pos2 = random.sample(range(seq_len), 2)
        
        input_vectors = []
        for idx in seq_indices:
            vec = [0.0] * (self.one_hot_dim + 1)
            vec[idx] = 1.0 
            input_vectors.append(vec)
            
        # SEP Token
        sep_vec = [0.0] * (self.one_hot_dim + 1)
        sep_vec[self.one_hot_dim] = 1.0
        input_vectors.append(sep_vec)
        
        cmd1_vec = self._int_to_binary_vec(pos1) + [0.0]
        input_vectors.append(cmd1_vec)
        cmd2_vec = self._int_to_binary_vec(pos2) + [0.0]
        input_vectors.append(cmd2_vec)
        
        target_seq = seq_indices[:]
        target_seq[pos1], target_seq[pos2] = target_seq[pos2], target_seq[pos1]
        
        target_tensor = torch.tensor(target_seq, dtype=torch.long)
        # target_onehot = F.one_hot(target_tensor, num_classes=self.one_hot_dim).float()
        
        input_tensor = torch.tensor(input_vectors, dtype=torch.float32)

        return (
            input_tensor,
            target_tensor,
            seq_len
        )
    
    def get_collate_fn(self):
        def collate_fn(batch):
            inputs, targets, lengths = zip(*batch)
            
            padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
            padded_targets = pad_sequence(targets, batch_first=True, padding_value=0.0)
            lengths = torch.tensor(lengths, dtype=torch.long)
            
            
            return padded_inputs, padded_targets, lengths
        return collate_fn
    
    def train(self, model, dataloader, optimizer, device):
        model.train()
        
        correct_list: list[float] = []
        total_loss: float = 0.0
        
        for _, batch in enumerate(dataloader):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            lengths = batch[2].to(device)
            
            output_lengths = (lengths - (inputs.shape[1] - targets.shape[1])).to(device)

            optimizer.zero_grad()
            out = model(x = inputs, lengths=lengths, out_lengths = targets.shape[1], train = True)
            loss, num_correct = model.compute_loss(
                out = out, 
                output_lengths=output_lengths, 
                targets = targets, 
                optimizer = optimizer
            )
            total_loss += loss.item()
            correct_list.append(num_correct)
            

        accuracy = sum(correct_list) / len(correct_list)
        
        return total_loss, accuracy
    
    @torch.inference_mode()
    def eval(self, model, dataloader, device):
        model.eval()
        correct_list: list[float] = []
        total_loss: float = 0.0
        
        
        
        for idx, batch in enumerate(dataloader):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            lengths = batch[2].to(device)
            
            length_diff = inputs.shape[1] - targets.shape[1]
            output_lengths = (lengths - length_diff).to(device)
            
            if idx == 0:
                flops = FlopCountAnalysis(model, (inputs, lengths, targets.shape[1], False))
                flops_1b = flops.total() / 1e9
            
            out = model(x = inputs, lengths=lengths, out_lengths = targets.shape[1], train = False)
            loss, num_correct = model.compute_loss(out = out, output_lengths=output_lengths, targets = targets)
            total_loss += loss.item()
            correct_list.append(num_correct)
            
        accuracy = sum(correct_list) / len(correct_list)
        
        return total_loss, accuracy, flops_1b