import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int,
        output_size: int,
        batch_size: int,
        job_type: str,
        device: str,
    ):
        super().__init__() 
        self.model_name: str = self.__class__.__name__
        
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size

        self.device: str = device
        
        self.rnn_cell = nn.RNNCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
        )
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
        batch_size, max_seq_len, _ = x.size()
        h_t = x.new_zeros(batch_size, self.hidden_size)
        

        masks = (torch.arange(max_seq_len).to(self.device) < lengths.unsqueeze(1)).float()
        outs = x.new_zeros(batch_size, out_lengths, self.output_size)
        
        for t in range(max_seq_len):
            mask = masks[:, t].unsqueeze(-1).repeat(1, self.hidden_size)
            x_t = x[:, t, :]
            next_h = self.rnn_cell(x_t, h_t)
            h_t = mask * next_h + (1 - mask) * h_t

        for idx in range(int(out_lengths)):
            x_t = x.new_zeros(batch_size, self.input_size)
            h_t = self.rnn_cell(x_t, h_t)
            out = self.output_layer(h_t)
            outs[:, idx, :] = out

        
        
        return outs, logs