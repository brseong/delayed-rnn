import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn import FlopCountAnalysis

IGNORE_IDX=-100   
INPUT_OUT_PADDING = 0.5
loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX)


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def compute_loss(out, model, output_lengths=None, targets=None, optimizer=None, device="cuda"):
    """
    Args:
        out (_type_): (batch_size, seq_len, num_classes) if output_lengths is not None else (batch_size, num_classes)
        output_lengths (_type_, optional): (batch_size,) if out is (batch_size, seq_len, num_classes) else None. Defaults to None.
        targets (_type_, optional): (batch, seq_len) if output_lengths is not None else (batch,). Defaults to None.
        optimizer (_type_, optional): optimizer. Defaults to None.

    Returns:
        loss: loss value
        corrects: number or rate of correct predictions in the batch
    """
    # Accuracy Calculation 
    if output_lengths is not None:
        _, max_seq_len, _ = out.size()
        masks = torch.arange(max_seq_len).unsqueeze(0).to(device) < output_lengths.unsqueeze(1) 
        out = out.masked_fill(~masks.unsqueeze(-1), IGNORE_IDX)  
        targets = targets.masked_fill(~masks, IGNORE_IDX)
    
    loss = loss_fn(out.transpose(1, 2), targets)
    
    
    predicted = out.argmax(dim=-1)
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
    # masking out of length positions
    is_correct = (predicted == targets).float()
    masked_correct = is_correct * masks
    token_acc_tensor = masked_correct.sum(dim=-1) / output_lengths
    seq_acc_tensor = torch.logical_or(predicted == targets, ~masks).all(dim=-1).float()
    
    if optimizer is not None:
        return loss, token_acc_tensor, seq_acc_tensor, total_grad_norm
    else:   
        return loss, token_acc_tensor, seq_acc_tensor


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