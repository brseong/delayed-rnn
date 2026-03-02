from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from math import log
import torch
import torch.nn as nn
from jaxtyping import Float, Int

from utils.config import Config, ModelType
from utils.random import clipped_gamma_sample

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus
from torch.nn.utils.rnn import pack_padded_sequence


presets={
    ModelType.RNN: Config(
        model_type=ModelType.RNN,
        # max_delay=40,
        max_think_steps=100,
        seed=None,
        batch_size=32,
        input_size=11,
        # seq_length=784,
        seq_min=5,
        seq_max=20,
        hidden_size=512,
        num_classes=10,
        learning_rate=0.01,
        epochs=100),
    ModelType.LSTM: Config(
        model_type=ModelType.LSTM,
        # max_delay=40,
        max_think_steps=100,
        seed=None,
        batch_size=32,
        input_size=11,
        # seq_length=784,
        seq_min=5,
        seq_max=20,
        hidden_size=256,
        num_classes=10,
        learning_rate=0.01,
        epochs=100),
    ModelType.GRU: Config(
        model_type=ModelType.GRU,
        # max_delay=40,
        max_think_steps=100,
        seed=None,
        batch_size=32,
        input_size=11,
        # seq_length=784,
        seq_min=5,
        seq_max=20,
        hidden_size=296,
        num_classes=10,
        learning_rate=0.01,
        epochs=100),
    ModelType.DelayedRNN: Config(
        model_type=ModelType.DelayedRNN,
        max_delay=40,
        max_think_steps=100,
        seed=None,
        batch_size=32,
        input_size=11,
        # seq_length=784,
        seq_min=5,
        seq_max=20,
        hidden_size=360,
        num_classes=10,
        learning_rate=0.01,
        epochs=100),
}

def get_model(model_class:ModelType, device:torch.device, config:Config|None=None) -> ThinkingRNN | ThinkingLSTM | FastThinkingLearnableDelayRNN | ThinkingGRU:
    if config is None:
        config = presets[model_class]
        config.device = device  # Set the device in the config for later use in model initialization
    match config.model_type:
        case ModelType.RNN:
            return ThinkingRNN(config.input_size, config.hidden_size, config.num_classes, config=config).to(config.device)
        case ModelType.LSTM:
            return ThinkingLSTM(config.input_size, config.hidden_size, config.num_classes, config=config).to(config.device)
        case ModelType.DelayedRNN:
            # return ThinkingLearnableDelayRNN(config.input_size, config.hidden_size, config.num_classes, max_delay=config.max_delay, config=config).to(config.device)
            return FastThinkingLearnableDelayRNN(config.input_size, config.hidden_size, config.num_classes, max_delay=config.max_delay, config=config).to(config.device)
        case ModelType.GRU:
            return ThinkingGRU(config.input_size, config.hidden_size, config.num_classes, config=config).to(config.device)
        case _: 
            raise ValueError(f"Unsupported model type: {config.model_type}")


@dataclass
class Seq2SeqOutput:
    outputs: Float[torch.Tensor, "batch_size N vocab_size"]
    # logits: Float[torch.Tensor, "batch_size N vocab_size"]
    think_steps: Int[torch.Tensor, "batch_size"]

class ThinkingRNN(nn.Module):
    def __init__(self, input_size:int, hidden_size: int, num_classes: int, config):
        super(ThinkingRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_think_steps = config.max_think_steps
        self.config = config
        
        # Set full vocabulary size by adding a 'think end' token to existing classes
        self.vocab_size = num_classes + 1 
        self.think_end_token = num_classes  # indices 0..num_classes-1 are normal classes, num_classes is the think-end token
        
        # Since the model receives one-hot encoded inputs, input_size equals vocab_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.vocab_size)
        self.device = config.device

    def forward(self, x, lengths, N=None, targets=None, teacher_forcing_ratio=0.0) -> Seq2SeqOutput:
        """
        Inference sequence generation function (batch processing optimized).
        x shape: (Batch, N+3, vocab_size) - one-hot encoded input sequence
        """
        batch_size = x.size(0)
        
        if N is None:
            N = x.size(1) - 3  
        
        # --- 1. Encoding stage (read the given N+3 sequence) ---
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        lengths_cpu = lengths.cpu() 
        packed_x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed_x, h0)
        
        # Skip padding and set each batch's true last token as the decoder's first input
        batch_indices = torch.arange(batch_size, device=self.device)
        dec_input = x[batch_indices, lengths - 1, :].unsqueeze(1)
        
        # Empty tensor to store final outputs of length N
        final_outputs = torch.zeros(batch_size, N, self.vocab_size, device=self.device)
        
        # Tensors to track per-sample state within the batch
        think_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        output_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        is_thinking = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        curr_input = dec_input
        
        # --- 2 & 3. Simultaneous batched processing for Thinking & Output ---
        while True:
            # Fully stop when all batches reach target output length N
            if (output_steps == N).all():
                break
                
            # [Core] Identify batches in output phase at this step (thinking finished, but fewer than N outputs produced)
            outputting_mask = (~is_thinking) & (output_steps < N)
            
            # Identify batches that are still thinking at this step
            thinking_mask = is_thinking.clone()
            
            # RNN forward pass (entire batch at once)
            out, h = self.rnn(curr_input, h)
            logits = self.fc(out.squeeze(1))  # (batch_size, vocab_size)
            one_hot_out = F.gumbel_softmax(logits, tau=1.0, hard=True) # (batch_size, vocab_size)
            
            # --- Output processing ---
            out_idx = torch.nonzero(outputting_mask, as_tuple=True)[0]
            if out_idx.numel() > 0:
                step_idx = output_steps[out_idx]
                
                # Write predicted tokens to the final output tensor
                final_outputs[out_idx, step_idx, :] = one_hot_out[out_idx]
                
                # Apply teacher forcing (None-check issue fixed)
                if targets is not None:
                    true_tokens = targets[out_idx, step_idx]
                    
                    # Decide teacher forcing only for non -1 (non-padding) targets
                    tf_rand = torch.rand(out_idx.numel(), device=self.device)
                    do_tf = (tf_rand < teacher_forcing_ratio) & (true_tokens >= 0)
                    
                    if do_tf.any():
                        tf_indices = out_idx[do_tf]
                        tf_step_idx = step_idx[do_tf]
                        tf_true_tokens = targets[tf_indices, tf_step_idx]
                        
                        # Overwrite next input with the ground-truth token for samples using teacher forcing
                        tf_one_hot = F.one_hot(tf_true_tokens, num_classes=self.vocab_size).float()
                        one_hot_out[tf_indices] = tf_one_hot
                
                # Increment output counter
                output_steps[outputting_mask] += 1
                next_input = one_hot_out.unsqueeze(1)
                
            # --- Thinking processing ---
            if thinking_mask.any():
                think_steps[thinking_mask] += 1
                
                pred_tokens = one_hot_out.argmax(dim=-1)
                hit_end = (pred_tokens == self.think_end_token)
                hit_max = (think_steps >= self.max_think_steps)
                
                # End thinking for a batch if think-end token is sampled or max thinking steps are exceeded
                just_finished = thinking_mask & (hit_end | hit_max)
                is_thinking[just_finished] = False
                next_input = logits.unsqueeze(1)  # During thinking, feed logits for richer gradients (can be changed to one_hot_out for more discrete behavior)
                
            # Prepare input for next step (contains ground-truth token where teacher forcing is applied)
            curr_input = next_input
            
        return Seq2SeqOutput(
            outputs=final_outputs[:, :, :-1], # Return only real-class probabilities, excluding think-end token
            think_steps=think_steps
        )

class ThinkingLSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size: int, num_classes: int, config):
        super(ThinkingLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_think_steps = config.max_think_steps
        self.config = config
        
        # Vocabulary size (number of classes + think-end token)
        self.vocab_size = num_classes + 1
        self.think_end_token = num_classes  
        self.device = config.device
        
        # Use LSTM: since input is one-hot, input_size = vocab_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.vocab_size)
    
    def forward(self, x, lengths, N=None, targets=None, teacher_forcing_ratio=0.0) -> Seq2SeqOutput:
        batch_size = x.size(0)
        
        if N is None:
            N = x.size(1) - 3  
        
        # --- 1. Encoding stage ---
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        
        lengths_cpu = lengths.cpu()
        packed_x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (h, c) = self.lstm(packed_x, (h0, c0))
        # [Fixed] Removed `_, (h, c) = self.lstm(x, (h0, c0))` from old code because it caused padding contamination.
        
        batch_indices = torch.arange(batch_size, device=self.device)
        dec_input = x[batch_indices, lengths - 1, :].unsqueeze(1)
        
        # Empty tensor for final outputs (keep logits for loss computation)
        final_outputs = torch.zeros(batch_size, N, self.vocab_size, device=self.device)
        
        # State-tracking tensors
        think_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        output_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        is_thinking = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        curr_input = dec_input
        
        # --- 2 & 3. Simultaneous batched processing for Thinking & Output ---
        while True:
            if (output_steps == N).all():
                break
                
            outputting_mask = (~is_thinking) & (output_steps < N)
            thinking_mask = is_thinking.clone()
            
            # LSTM forward pass (pass h and c as a tuple)
            out, (h, c) = self.lstm(curr_input, (h, c))
            logits = self.fc(out.squeeze(1))
            
            # Generate candidate next input with Gumbel-Softmax
            one_hot_out = F.gumbel_softmax(logits, tau=1.0, hard=True)
            
            # --- Output processing ---
            out_idx = torch.nonzero(outputting_mask, as_tuple=True)[0]
            if out_idx.numel() > 0:
                step_idx = output_steps[out_idx]
                
                # [Important] Store logits directly for CrossEntropyLoss
                final_outputs[out_idx, step_idx, :] = logits[out_idx]
                
                # Teacher Forcing
                if targets is not None:
                    true_tokens = targets[out_idx, step_idx]
                    tf_rand = torch.rand(out_idx.numel(), device=self.device)
                    do_tf = (tf_rand < teacher_forcing_ratio) & (true_tokens >= 0)
                    
                    if do_tf.any():
                        tf_indices = out_idx[do_tf]
                        tf_step_idx = step_idx[do_tf]
                        tf_true_tokens = targets[tf_indices, tf_step_idx]
                        
                        tf_one_hot = F.one_hot(tf_true_tokens, num_classes=self.vocab_size).float()
                        one_hot_out[tf_indices] = tf_one_hot
                
                output_steps[outputting_mask] += 1
                next_input = one_hot_out.unsqueeze(1)
                
            # --- Thinking processing ---
            if thinking_mask.any():
                think_steps[thinking_mask] += 1
                
                pred_tokens = one_hot_out.argmax(dim=-1)
                hit_end = (pred_tokens == self.think_end_token)
                hit_max = (think_steps >= self.max_think_steps)
                
                just_finished = thinking_mask & (hit_end | hit_max)
                is_thinking[just_finished] = False
                next_input = logits.unsqueeze(1)  # During thinking, feed logits for richer gradients (can be changed to one_hot_out for more discrete behavior)
                
            curr_input = next_input
            
        return Seq2SeqOutput(
            outputs=final_outputs[:, :, :-1], 
            think_steps=think_steps
        )

class ThinkingGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, config):
        super(ThinkingGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_think_steps = config.max_think_steps
        self.config = config
        
        # Vocabulary size (number of classes + think-end token)
        self.vocab_size = num_classes + 1
        self.think_end_token = num_classes  
        self.device = config.device
        
        # Use vocab_size as input_size to avoid internal dimension mismatch
        self.gru = nn.GRU(input_size=self.vocab_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, x, lengths, N=None, targets=None, teacher_forcing_ratio=0.0) -> Seq2SeqOutput:
        batch_size = x.size(0)
        
        if N is None:
            N = x.size(1) - 3  
        
        # --- 1. Encoding stage ---
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        
        lengths_cpu = lengths.cpu()
        packed_x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed_x, h0)
        
        batch_indices = torch.arange(batch_size, device=self.device)
        dec_input = x[batch_indices, lengths - 1, :].unsqueeze(1)  
        
        # Empty tensor for final outputs (keep logits format)
        final_outputs = torch.zeros(batch_size, N, self.vocab_size, device=self.device)
        
        # State-tracking tensors
        think_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        output_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        is_thinking = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        curr_input = dec_input
        
        # --- 2 & 3. Simultaneous batched processing for Thinking & Output ---
        while True:
            if (output_steps == N).all():
                break
                
            outputting_mask = (~is_thinking) & (output_steps < N)
            thinking_mask = is_thinking.clone()
            
            # GRU forward pass (only h is exchanged)
            out, h = self.gru(curr_input, h)
            logits = self.fc(out.squeeze(1))
            
            # Generate candidate next input with Gumbel-Softmax
            one_hot_out = F.gumbel_softmax(logits, tau=1.0, hard=True)
            
            # --- Output processing ---
            out_idx = torch.nonzero(outputting_mask, as_tuple=True)[0]
            if out_idx.numel() > 0:
                step_idx = output_steps[out_idx]
                
                # Store logits directly for loss computation
                final_outputs[out_idx, step_idx, :] = logits[out_idx]
                
                # Teacher Forcing
                if targets is not None:
                    true_tokens = targets[out_idx, step_idx]
                    tf_rand = torch.rand(out_idx.numel(), device=self.device)
                    do_tf = (tf_rand < teacher_forcing_ratio) & (true_tokens >= 0)
                    
                    if do_tf.any():
                        tf_indices = out_idx[do_tf]
                        tf_step_idx = step_idx[do_tf]
                        tf_true_tokens = targets[tf_indices, tf_step_idx]
                        
                        tf_one_hot = F.one_hot(tf_true_tokens, num_classes=self.vocab_size).float()
                        one_hot_out[tf_indices] = tf_one_hot
                
                output_steps[outputting_mask] += 1
                next_input = one_hot_out.unsqueeze(1)
                
            # --- Thinking processing ---
            if thinking_mask.any():
                think_steps[thinking_mask] += 1
                
                pred_tokens = one_hot_out.argmax(dim=-1)
                hit_end = (pred_tokens == self.think_end_token)
                hit_max = (think_steps >= self.max_think_steps)
                
                just_finished = thinking_mask & (hit_end | hit_max)
                is_thinking[just_finished] = False
                next_input = logits.unsqueeze(1)  # During thinking, feed logits for richer gradients (can be changed to one_hot_out for more discrete behavior)
                
            curr_input = next_input
            
        return Seq2SeqOutput(
            outputs=final_outputs[:, :, :-1], 
            think_steps=think_steps
        )

class FastThinkingLearnableDelayRNN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_classes:int, max_delay:int, config:Config):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_think_steps = config.max_think_steps
        self.config = config
        
        # If delay is 0 it is effectively a standard FFN, so it must be at least 1
        assert max_delay >= 1, "max_delay must be at least 1"
        self.max_delay = max_delay
        self.device = config.device
        
        self.vocab_size = num_classes + 1
        self.think_end_token = num_classes
        self.output_size = self.vocab_size
        
        self.afferent = nn.Linear(self.input_size, hidden_size)
        self.lateral = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(hidden_size, hidden_size))
        )
        self.efferent = nn.Linear(hidden_size, self.output_size)
        
        self.tau = nn.Parameter(max_delay * torch.rand_like(self.lateral) + 1)
        # self.scale_exponent = nn.Parameter(torch.zeros(hidden_size, device=self.device))
        self.scale_exponent = 0.5

    @staticmethod
    @torch.jit.script
    def calc_credit_matrix_jit(tau_clipped:torch.Tensor, max_delay:int, hidden_size:int, scale_exponent:float) -> torch.Tensor:
        # Same as previous version
        credit_matrix = torch.arange(max_delay + 1, out=tau_clipped.new_empty(max_delay + 1)) 
        credit_matrix = credit_matrix[:, None, None]
        distance = 1.0 + torch.abs(credit_matrix - tau_clipped)

        # raw_credit = distance.pow(-softplus(scale_exponent[None, :, None]))
        raw_credit = distance.pow(-scale_exponent)
        # differentiable = raw_credit
        # differentiable = raw_credit / (raw_credit.sum(dim=0, keepdim=True)) # Backward pass goes through this path
        logits = torch.log(raw_credit + 1e-8)
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
        perturbed_logits = logits + gumbel_noise
        differentiable = F.softmax(perturbed_logits, dim=0) # Forward and backward pass goes through this path
        
        # nondifferentiable = (credit_matrix == tau_clipped).float() # Forward pass uses this hard assignment for stability and interpretability
        max_idx = perturbed_logits.argmax(dim=0, keepdim=True)
        nondifferentiable = torch.zeros_like(logits).scatter_(0, max_idx, 1.0)
        
        # Straight-Through Estimator trick
        # Return equals a spike in the tau_clipped position,
        # but gradients flow through the differentiable soft assignment, enabling learning of tau.
        return differentiable + (nondifferentiable - differentiable).detach()
    
    @staticmethod
    @torch.jit.script
    def calc_credit_matrix_eval_jit(tau_clipped:torch.Tensor, max_delay:int, hidden_size:int, scale_exponent:float) -> torch.Tensor:
        # Same as previous version
        credit_matrix = torch.arange(max_delay + 1, out=tau_clipped.new_empty(max_delay + 1)) 
        credit_matrix = credit_matrix[:, None, None]
        
        nondifferentiable = (credit_matrix == tau_clipped).float() # Forward pass uses this hard assignment for stability and interpretability
        
        return nondifferentiable # No need for soft assignment during evaluation
    
    def calc_credit_matrix(self): 
        if self.training:
            return FastThinkingLearnableDelayRNN.calc_credit_matrix_jit(
                torch.clamp(self.tau, 1, self.max_delay)[None,...],
                self.max_delay,
                self.hidden_size,
                self.scale_exponent
            )
        else:
            return FastThinkingLearnableDelayRNN.calc_credit_matrix_eval_jit(
                torch.clamp(self.tau, 1, self.max_delay)[None,...],
                self.max_delay,
                self.hidden_size,
                self.scale_exponent
        )
    
    def _adjust_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Auto-pad/slice Gumbel-Softmax output (vocab_size) to input_size format."""
        if tensor.size(-1) != self.input_size:
            diff = self.input_size - tensor.size(-1)
            if diff > 0:
                tensor = F.pad(tensor, (0, diff))
            else:
                tensor = tensor[..., :self.input_size]
        return tensor

    def step_fast(self, x_t, history, ptr, W_rev):
        """
        [Core 1] Gather (pull) step function with O(1) memory write.
        x_t: (batch, input_size)
        history: (max_delay, batch, hidden_size) - past h_to_delay records
        W_rev: (max_delay, hidden_size, hidden_size) - reversed weight matrix
        """
        # 1. Lightly rotate weight matrix W by pointer position (much faster since there is no batch dimension)
        W_aligned = torch.roll(W_rev, shifts=ptr, dims=0)
        
        # 2. Multiply past history and weights, then aggregate delayed signals arriving at current step (Gather)
        # d: delay step, h: hidden_out, i: hidden_in, b: batch
        h_delayed = torch.einsum('dhi,dbi->bh', W_aligned, history)
        
        # 3. Compute new hidden state
        h_to_delay = torch.tanh(self.afferent(x_t) + h_delayed)
        
        # 4. Update buffer (use torch.where to avoid autograd in-place errors)
        mask = (torch.arange(self.max_delay, device=x_t.device) == ptr).view(-1, 1, 1)
        history = torch.where(mask, h_to_delay.unsqueeze(0), history)
        
        # 5. Move pointer and compute output
        ptr = (ptr + 1) % self.max_delay
        y_t = self.efferent(h_delayed)
        
        return history, ptr, y_t

    def forward(self, x, lengths, N=None, targets=None, teacher_forcing_ratio=0.0) -> Seq2SeqOutput:
        batch_size = x.size(0)
        device = x.device
        if N is None:
            N = x.size(1) - 3  

        # --- Pre-computation ---
        credit_matrix = self.calc_credit_matrix()
        # In the original code, credit_matrix[0] was cleared by (~mask) immediately after being read from buffer_ptr.
        # So it has no future effect, and slicing from [1:] is mathematically equivalent.
        W = credit_matrix[1:] * self.lateral[None, :, :]  # (max_delay, hidden, hidden)
        W_rev = W.flip(0)  # Pre-flip to match the ordering of past records
        
        # --- 1. Encoding stage ---
        history = x.new_zeros(self.max_delay, batch_size, self.hidden_size)
        ptr = 0 
        
        saved_history = history.new_zeros(history.size()) # (max_delay, batch_size, hidden_size)
        saved_ptrs = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            history, ptr, _ = self.step_fast(x_t, history, ptr, W_rev)
            
            is_last_token = (t == lengths - 1) 
            if is_last_token.any():
                mask = is_last_token.view(1, batch_size, 1) 
                saved_history = torch.where(mask, history, saved_history)
                saved_ptrs = torch.where(is_last_token, torch.tensor(ptr, device=device), saved_ptrs)
            
        # --- Alignment trick (synchronize pointers before Thinking starts) ---
        idx = torch.arange(self.max_delay, device=device).unsqueeze(1) # (max_delay, 1)
        gather_idx = (idx + saved_ptrs.unsqueeze(0)) % self.max_delay # (max_delay, batch_size)
        gather_idx = gather_idx.unsqueeze(-1).expand(self.max_delay, batch_size, self.hidden_size) # (max_delay, batch_size, hidden_size)
        
        # Align all batch histories so the pointer is unified to 0
        aligned_history = torch.gather(saved_history, 0, gather_idx) # (max_delay, batch_size, hidden_size)
        
        # --- 2. Thinking stage ---
        history = aligned_history
        ptr = 0  # Entire batch is now fully synchronized
        
        batch_indices = torch.arange(batch_size, device=device)
        curr_input = x[batch_indices, lengths - 1, :]
        
        is_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        think_steps_tensor = torch.zeros(batch_size, dtype=torch.long, device=device)
        think_steps = 0
        
        last_valid_input = curr_input.clone()
        
        while not is_done.all() and think_steps < self.max_think_steps:
            history, ptr, y_t = self.step_fast(curr_input, history, ptr, W_rev)
            
            curr_input_sampled = F.gumbel_softmax(y_t, tau=1.0, hard=True)
            curr_input_adj = self._adjust_dim(y_t)
            
            sampled_idx = curr_input_sampled.argmax(dim=-1)
            just_finished = (sampled_idx == self.think_end_token) & ~is_done
            
            # Cache the last valid token to use as first input of the output phase
            last_valid_input = torch.where((~is_done).unsqueeze(-1), curr_input_adj, last_valid_input)
            
            think_steps_tensor += (~is_done).long()
            is_done = is_done | just_finished
            think_steps += 1
            
            # [Core 2] Idling: inject zero input for finished batches (simulate SNN refractory period)
            curr_input = curr_input_adj * (~is_done).unsqueeze(-1).to(curr_input_adj.dtype)
            
        # --- 3. Output stage ---
        curr_input = last_valid_input
        final_outputs = x.new_zeros(batch_size, N, self.vocab_size)
        
        for j in range(N): 
            # curr_input: (batch, input_size) -> step_fast -> y_t: (batch, vocab_size)
            history, ptr, y_t = self.step_fast(curr_input, history, ptr, W_rev)
            final_outputs[:, j, :] = y_t
            
            # Model prediction (Batch, Vocab_size)
            one_hot_out = F.gumbel_softmax(y_t, tau=1.0, hard=True)
            
            # -----------------------------------------------------------------
            # [Updated batch-wise Teacher Forcing logic]
            # -----------------------------------------------------------------
            if targets is not None:
                # 1. Ground truth at step j (Batch,)
                targets_j = targets[:, j]
                
                # 2. Mask to check valid (non-padding, not -1) targets (Batch,)
                valid_mask = (targets_j >= 0)
                
                # 3. Teacher-forcing probability draw (Batch,) - sampled independently per sample
                tf_dice = (torch.rand(targets_j.size(0), device=targets.device) < teacher_forcing_ratio)
                
                # 4. Final mask for samples where teacher forcing is applied (Batch,)
                do_tf_mask = valid_mask & tf_dice
                
                # 5. One-hot encode targets (clamp -1 to 0 to avoid -1 index errors)
                # (safe because values where do_tf_mask is False are not used)
                safe_targets_j = torch.clamp(targets_j, min=0) # (Batch,)
                true_one_hot = F.one_hot(safe_targets_j, num_classes=self.vocab_size).float() # (Batch, Vocab_size)
                
                # 6. Match dimensions by expanding do_tf_mask from (Batch,) to (Batch, 1)
                do_tf_mask = do_tf_mask.unsqueeze(-1)
                
                # 7. Use torch.where to select ground truth (true_one_hot) or prediction (one_hot_out) by condition
                curr_input = torch.where(do_tf_mask, true_one_hot, one_hot_out) # (Batch, Vocab_size)
            else:
                # During inference, use only model predictions
                curr_input = one_hot_out
                
        return Seq2SeqOutput(outputs=final_outputs[:,:,:-1], think_steps=think_steps_tensor)

if __name__ == "__main__":
    # Example usage
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = get_model(ModelType.RNN, device=torch.device("cpu"), config=presets[ModelType.RNN])
    print("ThinkingRNN", count_parameters(model), "parameters")

    model = get_model(ModelType.LSTM, device=torch.device("cpu"), config=presets[ModelType.LSTM])
    print("ThinkingLSTM", count_parameters(model), "parameters")

    model = get_model(ModelType.GRU, device=torch.device("cpu"), config=presets[ModelType.GRU])
    print("ThinkingGRU", count_parameters(model), "parameters")
    
    model = get_model(ModelType.DelayedRNN, device=torch.device("cpu"), config=presets[ModelType.DelayedRNN])
    print("ThinkingLearnableDelayRNN", count_parameters(model), "parameters")