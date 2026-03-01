import torch
import torch.nn as nn
import torch.nn.functional as F

from models import compute_loss






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
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        assert self.max_delay > 0, "max_delay must be greater than 0"
        
        self.input_embedding = nn.Linear(self.input_size + self.hidden_size, self.hidden_size).to(self.device)
        self.tau_generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), self.sigmoid,
        ).to(self.device)
        self.mem_strength_generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), self.sigmoid,
        ).to(self.device)
        
        self.m_pass_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size)
        ).to(self.device)
        self.buffer_max_delay = self.max_delay + 1
        self.index_tensor = torch.arange(1, self.buffer_max_delay).to(self.device).unsqueeze(0).unsqueeze(0) # (1, 1, max_delay)
        if num_classes > 0:
            self.output_layer = nn.Linear(self.hidden_size, num_classes).to(self.device)
            
        
    
    def update_buffer(
        self, 
        hidden, 
        tau, 
        mem_l,
        buffer
    ):
        # tau: (batch_size, hidden_size)
        # hidden: (batch_size, hidden_size)
        # mem_l: (batch_size, hidden_size)
        new_buffer = buffer[:, :, 1:]
        
        tau = tau.unsqueeze(-1)  # (batch_size, hidden_size, 1)
        hidden_mem = hidden.unsqueeze(-1)  # (batch_size, hidden_size, 1)
        
        
        abs_distances = torch.abs(tau - self.index_tensor) # broadcasting
        intensity_weights = (mem_l.unsqueeze(-1) / (1 + abs_distances)) # (batch_size, hidden_size, max_delay)
        
        assert intensity_weights.max() <= 1.0 and intensity_weights.min() >= 0.0, "Intensity weights should be in the range [0, 1]"
        update_val = intensity_weights * hidden_mem
        new_buffer = new_buffer + update_val
        new_buffer = F.pad(new_buffer, (0, 1))
        
        return new_buffer
        
    def forward(self, x, lengths=None, out_lengths=None, train = False):
        """
        Args:
            masks : (batch_size, seq_len) - 0.0 for padding, 1.0 for valid tokens
            x_t: (batch_size, input_size)
            mask: (batch_size, 1)
            
        """
        logs: dict[str, float] = {}
        encode_tau_outs: list[torch.tensor] = []
        encode_mem_l_outs: list[torch.tensor] = []
        decode_tau_outs: list[torch.tensor] = []
        decode_mem_l_outs: list[torch.tensor] = []
        
        batch_size, max_seq_len, _ = x.size()
        
        masks = (torch.arange(max_seq_len).to(self.device) < lengths.unsqueeze(1)).float()
        buffer = torch.zeros(batch_size, self.hidden_size, self.buffer_max_delay).to(self.device)
        
        for idx in range(max_seq_len):
            x_t = x[:, idx, :]
            h_t = buffer[:, :, 0]
            mask = masks[:, idx].unsqueeze(-1) 
            x_h_t = torch.cat([h_t, x_t], dim=-1)
            h_t = self.input_embedding(x_h_t)
            passer_output = self.m_pass_layer(h_t)
            h_t = mask * passer_output + (1 - mask) * h_t
            
            tau = self.max_delay * self.tau_generator(h_t)
            mem_l = self.mem_strength_generator(h_t)
            tau = torch.clamp(tau, 1, self.max_delay) 
            
            # tau는 분포의 평균이 되는 기준점이므로 h_t가 각 delay slot에 어떤 값으로 저장될 지에 영향을 미친다. 
            buffer = self.update_buffer(
                hidden = h_t,
                tau = tau,
                mem_l = mem_l,
                buffer = buffer
            )
            encode_tau_outs.append(tau.mean().detach())
            encode_mem_l_outs.append(mem_l.mean().detach())
        
        outputs = []
        
        for _ in range(out_lengths):
            x_t = torch.zeros(batch_size, self.input_size).to(self.device) 
            h_t = buffer[:, :, 0]
            out = self.output_layer(h_t)
            outputs.append(out)
            
            x_h_t = torch.cat([h_t, x_t], dim=-1)
            h_t = self.input_embedding(x_h_t)
            h_t = self.m_pass_layer(h_t)
            
            tau = self.max_delay * self.tau_generator(h_t)
            mem_l = self.mem_strength_generator(h_t)
            tau = torch.clamp(tau, 1, self.max_delay)
            
            buffer = self.update_buffer(
                hidden = h_t,
                tau = tau,
                mem_l = mem_l,
                buffer = buffer
            )
            decode_tau_outs.append(tau.mean().detach())
            decode_mem_l_outs.append(mem_l.mean().detach())
            
        out = torch.stack(outputs, dim=1) # (batch_size, out_seq_len, num_classes)
        
        if torch.jit.is_tracing():
                return out
        
        # enc_len = len(encode_tau_outs)
        # logs["encode_tau_mean"] = sum(encode_tau_outs) / enc_len
        # logs["encode_mem_l_mean"] = sum(encode_mem_l_outs) / enc_len
        # logs["encode_tau_var"] = sum((t - logs["encode_tau_mean"]) ** 2 for t in encode_tau_outs) / enc_len
        # logs["encode_mem_l_var"] = sum((m - logs["encode_mem_l_mean"]) ** 2 for m in encode_mem_l_outs) / enc_len
    
        # dec_len = len(decode_tau_outs)
        # logs["decode_tau_mean"] = sum(decode_tau_outs) / dec_len
        # logs["decode_mem_l_mean"] = sum(decode_mem_l_outs) / dec_len
        # logs["decode_tau_var"] = sum((t - logs["decode_tau_mean"]) ** 2 for t in decode_tau_outs) / dec_len
        # logs["decode_mem_l_var"] = sum((m - logs["decode_mem_l_mean"]) ** 2 for m in decode_mem_l_outs) / dec_len

        logs["encode_tau_mean"] = torch.stack(encode_tau_outs).mean().item()
        logs["encode_mem_l_mean"] = torch.stack(encode_mem_l_outs).mean().item()
        logs["encode_tau_var"] = torch.stack(encode_tau_outs).var().item()
        logs["encode_mem_l_var"] = torch.stack(encode_mem_l_outs).var().item()  
        
        logs["decode_tau_mean"] = torch.stack(decode_tau_outs).mean().item()
        logs["decode_mem_l_mean"] = torch.stack(decode_mem_l_outs).mean().item()
        logs["decode_tau_var"] = torch.stack(decode_tau_outs).var().item()
        logs["decode_mem_l_var"] = torch.stack(decode_mem_l_outs).var().item()
        
        return out, logs