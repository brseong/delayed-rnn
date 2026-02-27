import torch
import torch.nn as nn 
import torch.nn.functional as F

from models import compute_loss, INPUT_OUT_PADDING

class DelayBuffer:
    def __init__(
        self, 
        max_delay, 
        batch_size, 
        input_size,
        device
    ):
        super().__init__()
        self.max_delay: int = max_delay
        self.input_size: int = input_size
        
        self.device: str = device
        """
        delay된 입력 벡터들을 하나의 벡터로 인코딩하는 법: Circular Orthogonal Basis
        """
        self.buffer = torch.zeros(batch_size, max_delay + 1, max_delay, input_size).to(torch.device(self.device))
        self.decay = 0.99
        
    def buffer_update(self, input, delay_gate_output):
        """
        input: [batch_size, input_size]
        delay_gate_output: [batch_size, max_delay]
        """
        new_buffer = torch.zeros_like(self.buffer)
        
        new_buffer[:, :-1, :, :] = self.buffer[:, 1:, :, :]
        tau_indices = torch.arange(1, self.max_delay + 1, device=self.device).float()
        decay_coeffs = (self.decay ** tau_indices).view(1, -1, 1) # [1, max_delay, 1]
        update_values = input.unsqueeze(1) * decay_coeffs * delay_gate_output.unsqueeze(-1)
        
        d_indices = torch.arange(self.max_delay, device=self.device)
        new_buffer[:, d_indices, d_indices, :] = update_values
        self.buffer = new_buffer
        
    def get_current_memory(self, slide = False):
        if slide:
            self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
            self.buffer[:, -1, :, :] = 0.0 
            
        mem = self.buffer[:, 0, :, :]
        mem_t = mem.sum(dim=1) # [batch_size, max_delay, input_size]
        return mem_t
    
    def reset(self, batch_size):
        self.buffer = torch.zeros(batch_size, self.max_delay + 1, self.max_delay, self.input_size).to(self.device)
        
class DelayRNN(nn.Module):
    def __init__(
        self,
        max_delay: int,
        num_layers: int,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        batch_size: int,
        is_classification: bool = True,
        device: str = "cuda",
    ):
        self.model_name: str = self.__class__.__name__
        super().__init__()
        self.max_delay: int = max_delay
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.num_classes: int = num_classes
        
        self.device: str = device
        self.is_classification: bool = is_classification
        self.compute_loss = compute_loss
        
        """
        잠재적 문제:
            1. delay gate가 모든 step에 대해 0.5 이상의 값을 출력하는 경우, DelayBuffer의 값이 너무 커질 수 있음 (특히 긴 시퀀스에서)
        """
        self.delay_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, self.max_delay), nn.ELU(),
            nn.Linear(self.max_delay, self.max_delay),
        ).to(self.device)
        
        self.DelayBuffer = DelayBuffer(
            max_delay = self.max_delay,
            batch_size = batch_size,  
            input_size = self.input_size,
            device = self.device
        )
        
        self.input_embedding = nn.Sequential(
            nn.Linear(input_size * 2, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, hidden_size)
        ).to(self.device)
        
        self.rnn_cell = nn.GRUCell(hidden_size, hidden_size).to(self.device)
        
        if self.num_classes > 0:
            self.fc = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.num_classes)
            ).to(self.device)

    # def out2input(self, out):
    #     dec_input = F.gumbel_softmax(out, hard=True)
    #     dec_input = F.pad(dec_input, (0, self.input_size - self.num_classes), value=INPUT_OUT_PADDING)
    #     return dec_input
    
    # def delay_gate_ste(self, gate_out):
    #     gate_out_ste = (gate_out > 0.5).float()
    #     return gate_out + (gate_out_ste - gate_out).detach()
        
    def forward(self, x, lengths=None, out_lengths=None, train = False):
        batch_size, max_seq_len, _ = x.size()
        
        self.DelayBuffer.reset(batch_size = batch_size)
        
        h_t = torch.zeros(batch_size, self.hidden_size).to(self.device)

        masks = (torch.arange(max_seq_len).to(self.device) < lengths.unsqueeze(1)).float()
        for t in range(max_seq_len):
            x_t = x[:, t, :]
            mask = masks[:, t].unsqueeze(-1)
            cell_input = torch.cat([x_t, h_t], dim=-1)
            mem_t = self.DelayBuffer.get_current_memory()
            delay_gate_output = F.gumbel_softmax(self.delay_gate(cell_input), hard=True) 
            self.DelayBuffer.buffer_update(x_t, delay_gate_output)
            
            
            rnn_input = torch.cat([x_t, mem_t], dim=-1)
            rnn_input = self.input_embedding(rnn_input)
            next_h = self.rnn_cell(rnn_input, h_t)
            h_t = mask * next_h + (1 - mask) * h_t
            
        outputs = []
        
        for t in range(out_lengths):
            x_t = torch.zeros(batch_size, self.input_size).to(self.device)
            mem_t = self.DelayBuffer.get_current_memory(slide = True)
            cell_input = torch.cat([x_t, h_t], dim=-1)
            delay_gate_output = F.gumbel_softmax(self.delay_gate(cell_input), hard=True)
            self.DelayBuffer.buffer_update(x_t, delay_gate_output)
            
            rnn_input = torch.cat([x_t, mem_t], dim=-1)
            rnn_input = self.input_embedding(rnn_input)
            next_h = self.rnn_cell(rnn_input, h_t)
            
            if self.num_classes > 0:
                out = self.fc(next_h)
                outputs.append(out.unsqueeze(1))
                
            h_t = next_h
            
            
        out = torch.cat(outputs, dim=1) # (batch_size, target_seq_len, num_classes)
        return out 