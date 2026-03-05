import torch 
import math
import torch.nn as nn 
from model import PositionalEncoding

class Transformer(nn.Module):
    def __init__(
        self,
        nhead: int, 
        encoding_method: str, 
        input_size: int,
        hidden_size: int,
        output_size: int,
        batch_size: int,
        job_type: str,
        device: str,
    ):
        super().__init__()
        self.model_name: str = self.__class__.__name__
        self.device: str = device
        self.job_type: str = job_type
        self.encoding_method: str = encoding_method
        
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size
        self.batch_size: int = batch_size
        
        self.embedding = nn.Linear(input_size, self.hidden_size).to(self.device)
        if self.encoding_method == 'positional':
            self.time_encoder = PositionalEncoding(d_model = self.hidden_size).to(self.device)
        else:
            raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, 
            nhead=nhead, 
            dim_feedforward=self.hidden_size, 
            batch_first=True 
        )
        
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=nhead,
            dim_feedforward=self.hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.output_size)
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
        
        embedded = self.embedding(x)
        embedded = self.time_encoder(embedded)
        
        encoder_masks = (torch.arange(x.size(1)).to(self.device) < lengths.unsqueeze(1)).float()

        embedded = self.encoder_layer(embedded, src_key_padding_mask=~encoder_masks.bool())
            
        decoder_input = torch.zeros(batch_size, out_lengths, self.hidden_size).to(self.device)
        decoder_input = self.time_encoder(decoder_input)
        tgt_mask = torch.triu(torch.ones(out_lengths, out_lengths, device=self.device) * float('-inf'), diagonal=1)
        
        hidden = self.decoder_layer(
            decoder_input, 
            embedded, 
            tgt_key_padding_mask=None, 
            tgt_mask = tgt_mask,
            memory_key_padding_mask=~encoder_masks.bool()
        )
        
        outs = self.fc(hidden)
        return outs, logs