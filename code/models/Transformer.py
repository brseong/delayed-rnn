import torch 
import math
import torch.nn as nn 
from models import compute_loss, PositionalEncoding
    
class Transformer(nn.Module):
    def __init__(
        self,
        nhead: int, 
        encoding_method: str, 
        input_size: int,  
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        batch_size: int,
        is_classification: bool = True,
        device: str = "cuda",

    ):
        super().__init__()
        self.model_name: str = self.__class__.__name__
        self.device: str = device
        self.encoding_method: str = encoding_method
        
        self.hidden_size: int = hidden_size
        self.num_classes: int = num_classes
        self.is_classification = is_classification
        self.batch_first = True
        self.compute_loss = compute_loss
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        self.embedding = nn.Linear(input_size, self.hidden_size).to(self.device)
        if self.encoding_method == 'positional':
            self.time_encoder = PositionalEncoding(d_model = self.hidden_size).to(self.device)
        else:
            raise ValueError(f"Unsupported encoding method: {self.encoding_method}")
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size, 
                nhead=nhead, 
                dim_feedforward=self.hidden_size, 
                batch_first=self.batch_first 
            )
            for _ in range(num_layers)
        ]).to(self.device)
        
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=nhead,
                dim_feedforward=self.hidden_size,
                batch_first=self.batch_first
            ).to(self.device)
            for _ in range(num_layers)
        ]).to(self.device)
        
        if self.num_classes > 0:
            self.fc = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.num_classes)
            ).to(self.device)
        
    def forward(self, x, lengths=None, out_lengths=None, train=False):
        batch_size, _, _ = x.size()
        # 1. Embedding & Position Encoding
        # sqrt: scaling foctor for stable training because of the adding of positional encoding
        x = self.embedding(x) * math.sqrt(self.hidden_size) 
        x = self.time_encoder(x)
        
        if lengths is not None:
            encoder_masks = (torch.arange(x.size(1)).to(self.device) < lengths.unsqueeze(1)).float()
            for layer in self.encoder_layers:
                x = layer(x, src_key_padding_mask=(encoder_masks == 0))

            de_input = torch.zeros(batch_size, 1, self.hidden_size, device=self.device)
            outputs = []
            for _ in range(out_lengths):
                de_input_with_pe = self.time_encoder(de_input)
                sz = de_input_with_pe.size(1)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(sz, device=self.device)
                for layer in self.decoder_layers:
                    decoder_output = layer(
                        de_input_with_pe, 
                        x, 
                        tgt_key_padding_mask=None, 
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=(encoder_masks == 0)
                    )
                last_decoder_hidden = decoder_output[:, -1:, :]
                de_input = torch.cat([de_input, last_decoder_hidden], dim=1)
                last_decoder_output = self.fc(last_decoder_hidden)
                outputs.append(last_decoder_output)
                
            
            out = torch.cat(outputs, dim=1)
            
        return out, None