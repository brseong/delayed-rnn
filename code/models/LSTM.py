import torch
import torch.nn as nn
from models import compute_loss

class LSTM(nn.Module):
    def __init__(
        self, 
        num_layers: int, 
        input_size: int, 
        hidden_size: int, 
        num_classes: int,  
        device: str = "cuda",
        is_classification: bool = True
    ):
        self.model_name: str = self.__class__.__name__
        super().__init__() 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.device = device
        self.is_classification = is_classification
        
        self.compute_loss = compute_loss
        
        self.lstm_cell = nn.LSTMCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
        ).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        if self.num_classes > 0:
            self.fc = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.num_classes)
            ).to(self.device)
      
    def forward(self, x, lengths=None, out_lengths=None, train = False):
        
        batch_size, max_seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(self.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=self.device)
        state = (h_t, c_t)
        
        if lengths is not None:
            """_summary_

            Args:
                x : (batch_size, seq_len + 3, input_size + 1)
                lengths : (batch_size,) if length is not None
                targets : (batch_size, seq_len, input_size) 
            """
            masks = masks = (torch.arange(max_seq_len).to(self.device) < lengths.unsqueeze(1)).float()
            outputs = []
            
            for t in range(max_seq_len):
                mask = masks[:, t].unsqueeze(-1).repeat(1, self.hidden_size)
                x_t = x[:, t, :]
                next_h, next_c = self.lstm_cell(x_t, state)
                
                h_t = mask * next_h + (1 - mask) * state[0]
                c_t = mask * next_c + (1 - mask) * state[1]
                state = (h_t, c_t) 
                
            for t in range(out_lengths):
                x_t = torch.zeros(batch_size, self.input_size).to(self.device) 
                next_h, next_c = self.lstm_cell(x_t, state)
                out = self.fc(next_h)
                outputs.append(out.unsqueeze(1))
                
                state = (next_h, next_c)
            
            out = torch.cat(outputs, dim=1) # (batch_size, target_seq_len, num_classes)
            
        else:
            for t in range(max_seq_len):
                x_t = x[:, t, :]
                state = self.lstm_cell(x_t, state)
            h_t, _ = state
            out = self.fc(h_t) # (batch_size, num_classes)
            
        return out 