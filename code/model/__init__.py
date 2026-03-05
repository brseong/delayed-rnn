import math
import torch
import torch.nn as nn

from utils import (
    NS_seq2seq
)

IGNORE_IDX=-100000
LOSS_FN = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX)

def compute_loss(
    out,
    model,
    out_lengths,
    targets,
    optimizer,
    job_type,
    dataset_name,
    logs
):
    _, max_seq_len, _ = out.size()
    masks = torch.arange(max_seq_len).unsqueeze(0).to(out.device) < out_lengths.unsqueeze(1)
    out = out.masked_fill(~masks.unsqueeze(-1), IGNORE_IDX)
    
    # If targets is longer than out, we need to trim the targets to match the length of out.
    if targets.size(1) > out.size(1):
        targets = targets[:, :out.size(1)]
    
    targets = targets.masked_fill(~masks, IGNORE_IDX)
        
    loss = LOSS_FN(out.transpose(1, 2), targets)
    predicted = out.argmax(dim=-1)
    
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        
    is_correct = (predicted == targets).float()
    masked_correct = is_correct * masks
    token_acc_tensor = (masked_correct.sum(dim=-1) / out_lengths).mean().item()
    seq_acc_tensor = (torch.logical_or(predicted == targets, ~masks).all(dim=-1).float()).mean().item()
    
    logs["loss"] = loss.item()
    logs["token_acc"] = token_acc_tensor
    logs["seq_acc"] = seq_acc_tensor
    if optimizer is not None:
        logs["total_grad_norm"] = total_grad_norm.item() 
        
        
    return logs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x