import torch
import torch.nn as nn
import torch.nn.functional as F

def update_buffer(
    message,
    tau,
    mem_i, 
    buffer,
    index_tensor,
    mask
):
    new_buffer = F.pad(buffer[:, :, 1:], (0, 1)) 
    abs_distances = torch.abs(tau - index_tensor)
    
    intensity_weights = mem_i * (1- abs_distances) ** 2
    
    assert intensity_weights.max() <= 1.0 and intensity_weights.min() >= 0.0, "Intensity weights should be in the range [0, 1]" 
    
    new_buffer = (1 - intensity_weights) * new_buffer + intensity_weights * message
    if mask is not None:
        return torch.where(mask.bool(), new_buffer, buffer) # B x H x MaxDelay
    else:
        return new_buffer

class DelayRNN(nn.Module):
    def __init__(
        self,
        max_delay: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        batch_size: int,
        job_type: str,
        device: str,
    ):
        """ DelayRNN model which incorporates a memory buffer with learnable intensity weights based on the distance of past hidden states.
        Args:
            max_delay (int): Maximum delay for the memory buffer.
            input_size (int): Input feature dimension.
            hidden_size (int): Hidden state dimension.
            output_size (int): Output feature dimension (number of classes for classification).
            batch_size (int): Batch size for training.
            job_type (str): Type of task.
            device (str): Device to run the model on (e.g., "cuda" or "cpu").
        """
        super().__init__()
        self.max_delay: int = max_delay
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.half_hidden_size: int = hidden_size // 2
        self.output_size: int = output_size
        self.batch_size: int = batch_size
        
        self.job_type: str = job_type
        self.device: str = device
        self.model_name: str = self.__class__.__name__
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() 
        
        self.message_generator = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size), self.tanh
        )
        self.tau_generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.half_hidden_size), self.relu,
            nn.Linear(self.half_hidden_size, 2 * self.hidden_size), self.sigmoid
        ) # information crossing between tau and mem_i is possible/
        
        self.register_buffer("index_tensor", torch.arange(0, self.max_delay).view(1, 1, -1) / self.max_delay)
        
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        self.to(self.device)
        
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor=None, 
        out_lengths: int=None, 
        train: bool=False, 
        targets: torch.Tensor=None,
        logs: dict[str, torch.Tensor] = {}
    ):
        batch_size, max_input_len, _ = x.size()
        
        # mean / var of tau and mem_l across time steps for analysis
        enc_tau_all = x.new_zeros((2, max_input_len))
        enc_message_all = x.new_zeros((2, max_input_len))
        dec_tau_all = x.new_zeros((2, out_lengths))
        dec_message_all = x.new_zeros((2, out_lengths))
        
        arange = torch.arange(max_input_len).to(self.device)
        masks = (arange < lengths.unsqueeze(1)).to(x.dtype)
        buffer = x.new_zeros(batch_size, self.hidden_size, self.max_delay)
        
        for idx in range(max_input_len):
            x_t = x[:, idx, :]
            m_t = buffer[:, :, 0]
            mask = masks[:, idx].unsqueeze(-1) # B x 1
            x_m_t = torch.cat([x_t, m_t], dim=-1)

            message_t = mask * self.message_generator(x_m_t)
            tau = self.tau_generator(message_t)[:, :self.hidden_size]
            mem_i =  self.tau_generator(message_t)[:, self.hidden_size:]
            
            buffer = update_buffer(
                message = message_t.unsqueeze(-1),
                tau = tau.unsqueeze(-1),
                mem_i = mem_i.unsqueeze(-1),
                buffer = buffer,
                index_tensor = self.index_tensor,
                mask = mask.bool().unsqueeze(-1).expand(-1, self.hidden_size, -1)
            )
            
            enc_tau_all[0, idx] = tau.detach().mean()
            enc_tau_all[1, idx] = tau.detach().std()
            enc_message_all[0, idx] = message_t.detach().mean()
            enc_message_all[1, idx] = message_t.detach().std()
        
        x_t = x.new_zeros((batch_size, self.input_size))
        outs = x.new_zeros(batch_size, out_lengths, self.output_size)
        
        for idx in range(out_lengths):
            m_t = buffer[:, :, 0]
            x_m_t = torch.cat([x_t, m_t], dim=-1)
            message_t = self.message_generator(x_m_t)
            tau = self.tau_generator(message_t)[:, :self.hidden_size]
            mem_i = self.tau_generator(message_t)[:, self.hidden_size:]
            
            buffer = update_buffer(
                message = message_t.unsqueeze(-1),
                tau = tau.unsqueeze(-1),
                mem_i = mem_i.unsqueeze(-1),
                buffer = buffer,
                index_tensor = self.index_tensor,
                mask = None,
            )
            
            out = self.output_layer(message_t)
            outs[:, idx, :] = out
            
            dec_tau_all[0, idx] = tau.detach().mean()
            dec_tau_all[1, idx] = tau.detach().std()
            dec_message_all[0, idx] = message_t.detach().mean()
            dec_message_all[1, idx] = message_t.detach().std()
        
        logs["model/encode_tau_mean"] = enc_tau_all[0].mean()
        logs["model/encode_tau_std"] = enc_tau_all[1].mean()
        logs["model/encode_message_mean"] = enc_message_all[0].mean()
        logs["model/encode_message_std"] = enc_message_all[1].mean()
        
        logs["model/decode_tau_mean"] = dec_tau_all[0].mean()
        logs["model/decode_tau_std"] = dec_tau_all[1].mean()
        logs["model/decode_message_mean"] = dec_message_all[0].mean()
        logs["model/decode_message_std"] = dec_message_all[1].mean()
        
        return outs, logs
            
            
            
            
        
    