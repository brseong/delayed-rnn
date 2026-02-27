import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn import FlopCountAnalysis

IGNORE_IDX=-100   
loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX)

def compute_loss(out, output_lengths=None, targets=None, optimizer=None, device="cuda"):
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
        optimizer.step()
        
    if output_lengths is not None:
        # masking out of length positions
        is_correct = (predicted == targets).float()
        masked_correct = is_correct * masks
        corrects = masked_correct.sum() / output_lengths.sum()
    else:
        # number of correct predictions in the batch
        corrects = (predicted == targets).sum().item() 

    return loss, corrects