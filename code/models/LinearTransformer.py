import torch 
import math
import torch.nn as nn 
import torch.nn.functional as F

from models import compute_loss, PositionalEncoding

class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, q, k, v, mask=None, prev_state=None):
        q = F.elu(q) + 1 
        k = F.elu(k) + 1

        if mask is not None:
            k = k * mask
            v = v * mask

        S_prev, z_prev = prev_state
        
        S_t = S_prev + torch.einsum("bhsd,bhsm->bhdm", k, v)
        z_t = z_prev + k.transpose(-2, -1) # [B, H, D, 1]
        
        num = torch.einsum("bhsd,bhdm->bhsm", q, S_t)
        den = torch.einsum("bhsd,bhdm->bhs", q, z_t).unsqueeze(-1)
        
        return num / (den + self.eps), (S_t, z_t)

class LinearTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = LinearAttention()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x, mask=None, prev_state=None):
        B, S, E = x.shape
        qkv = self.qkv_proj(x).chunk(3, dim=-1) # qkv: tuple of 3 tensors, each [B, S, E]
        q, k, v = map(lambda t: t.view(B, S, self.nhead, self.d_head).transpose(1, 2), qkv) # q, k, v: [B, H, S, D]
        
        # attention에서 새 state를 반환받음
        attn_out, new_state = self.attention(q, k, v, mask=mask, prev_state=prev_state)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, E)
        
        x = self.norm1(x + self.out_proj(attn_out))
        x = self.norm2(x + self.ff(x))
        return x, new_state

class LinearTransformer(nn.Module):
    def __init__(self, 
        nhead, 
        encoding_method, 
        input_size, 
        hidden_size, 
        num_layers, 
        num_classes: int, 
        batch_size: int,
        is_classification: bool = True,
        device: str = "cuda",
    ):
        self.model_name: str = self.__class__.__name__
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.d_head = hidden_size // nhead
        self.encoding_method = encoding_method
        self.num_classes: int = num_classes
        self.is_classification: bool = is_classification
        
        self.embedding = nn.Linear(input_size, hidden_size).to(self.device)
        if self.encoding_method == 'positional':
            self.time_encoder = PositionalEncoding(d_model = self.hidden_size).to(self.device)
        else:
            raise ValueError(f"Unsupported encoding method: {self.encoding_method}")
        self.compute_loss = compute_loss
        
        self.encode_layers = nn.ModuleList([
            LinearTransformerLayer(hidden_size, nhead, hidden_size) for _ in range(num_layers)
        ]).to(self.device)
        self.decode_layers = nn.ModuleList([
            LinearTransformerLayer(hidden_size, nhead, hidden_size) for _ in range(num_layers)
        ]).to(self.device)
        
        if self.num_classes > 0:
            self.fc = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.num_classes)
            ).to(self.device)
            
    def forward(self, x, lengths=None, out_lengths=None, train = False):
        batch_size, max_input_len, _ = x.size()
        x = self.embedding(x)
        if self.encoding_method == 'positional':
            x = self.time_encoder(x)
            
        mask = (torch.arange(x.size(1)).to(self.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        mask = mask.unsqueeze(1).repeat(1, self.nhead, 1).unsqueeze(-1).to(self.device) 
        
        init_s = torch.zeros(batch_size, self.nhead, self.d_head, self.d_head, device=self.device)
        init_z = torch.zeros(batch_size, self.nhead, self.d_head, 1, device=self.device)
        
        states = [(init_s.clone(), init_z.clone()) for _ in range(self.num_layers)]
        for t in range(max_input_len):
            x_t = x[:, t:t+1, :]
            for i, layer in enumerate(self.encode_layers):
                x_t, states[i] = layer(x_t, mask=mask[:, :, t:t+1, :], prev_state=states[i])
        
        outputs = []
        de_input = x_t
        for t in range(out_lengths): 
            for i, layer in enumerate(self.decode_layers):
                de_output, states[i] = layer(de_input, prev_state=states[i])
            
            if self.num_classes > 0:
                output = self.fc(de_output)
                outputs.append(output)
            
            de_input = de_output
            
        out = torch.cat(outputs, dim=1)
        
        return out