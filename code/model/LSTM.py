import torch
import torch.nn as nn

class LSTM(nn.Module):
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
        
        self.lstm_cell = nn.LSTMCell(
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
        c_t = x.new_zeros(batch_size, self.hidden_size)
        state = (h_t, c_t)

        masks = (torch.arange(max_seq_len).to(self.device) < lengths.unsqueeze(1)).float()
        outs = x.new_zeros(batch_size, out_lengths, self.output_size)
        
        for t in range(max_seq_len):
            mask = masks[:, t].unsqueeze(-1)
            x_t = x[:, t, :]
            next_h, next_c = self.lstm_cell(x_t, state)
            state = (mask * next_h + (1 - mask) * state[0], mask * next_c + (1 - mask) * state[1])

        for idx in range(int(out_lengths)):
            x_t = x.new_zeros(batch_size, self.input_size)
            next_h, next_c = self.lstm_cell(x_t, state)
            out = self.output_layer(next_h)
            outs[:, idx, :] = out
            
            state = (next_h, next_c)
        
        
        return outs, logs