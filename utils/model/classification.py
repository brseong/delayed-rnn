from typing import Any

import torch
import torch.nn as nn
import math

from jaxtyping import Float
from utils.config import Config, ModelType


presets={
    ModelType.RNN: Config(model_type=ModelType.RNN,
        max_delay=20,
        # max_think_steps=100,
        seed=None,
        batch_size=32,
        input_size=1,
        seq_length=784,
        # seq_min=5,
        # seq_max=20,
        hidden_size=128,
        num_classes=10,
        learning_rate=0.001,
        epochs=100,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ),
    ModelType.LSTM: Config(model_type=ModelType.LSTM,
        max_delay=20,
        # max_think_steps=100,
        seed=None,
        batch_size=32,
        input_size=1,
        seq_length=784,
        # seq_min=5,
        # seq_max=20,
        hidden_size=64,
        num_classes=10,
        learning_rate=0.001,
        epochs=100,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ),
    ModelType.GRU: Config(model_type=ModelType.GRU,
        max_delay=20,
        # max_think_steps=100,
        seed=None,
        batch_size=32,
        input_size=1,
        seq_length=784,
        # seq_min=5,
        # seq_max=20,
        hidden_size=75,
        num_classes=10,
        learning_rate=0.005,
        epochs=100,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ),
    ModelType.Transformer: Config(model_type=ModelType.Transformer,
        max_delay=20,
        # max_think_steps=100,
        seed=None,
        batch_size=32,
        input_size=1,
        seq_length=784,
        # seq_min=5,
        # seq_max=20,
        hidden_size=32,
        num_classes=10,
        learning_rate=0.001,
        epochs=100,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ),
    ModelType.DelayedRNN: Config(model_type=ModelType.DelayedRNN,
        max_delay=20,
        # max_think_steps=100,
        seed=None,
        batch_size=32,
        input_size=1,
        seq_length=784,
        # seq_min=5,
        # seq_max=20,
        hidden_size=90,
        num_classes=10,
        learning_rate=0.001,
        epochs=100,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ),
}

def get_model_with_preset(model_class:ModelType) -> nn.Module:
    config = presets[model_class]
    match config.model_type:
        case ModelType.RNN:
            return SimpleRNN(config.input_size, config.hidden_size, config.num_classes, config).to(config.device)
        case ModelType.LSTM:
            return SimpleLSTM(config.input_size, config.hidden_size, config.num_classes, config).to(config.device)
        case ModelType.GRU:
            return SimpleGRU(config.input_size, config.hidden_size, config.num_classes, config).to(config.device)
        case ModelType.Transformer:
            return SimpleTransformer(config.input_size, config.hidden_size, config.num_classes, config).to(config.device)
        case ModelType.DelayedRNN:
            return LearnableDelayRNN(config.input_size, config.hidden_size, config.num_classes, config.max_delay, config).to(config.device)
        case _:
            raise ValueError(f"Unsupported model type: {config.model_type}")



class SimpleRNN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_classes:int, config:Config):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        # batch_first=True: 입력 형태를 (Batch, Seq, Feature)로 받음
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = config.device
        self.config = config
    def forward(self, x):
        # x shape: (Batch, 784, 1)
        
        # RNN 초기 은닉 상태 (0으로 초기화)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
        
        # RNN 순전파
        # out shape: (Batch, Seq_Length, Hidden_Size)
        out, _ = self.rnn(x, h0)
        
        # 마지막 타임스텝(Time step)의 결과만 가져와서 분류
        # out[:, -1, :] shape: (Batch, Hidden_Size)
        out = self.fc(out[:, -1, :])
        return out

class SimpleLSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_classes:int, config:Config):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.device = config.device
        self.config = config
        # batch_first=True: 입력 형태를 (Batch, Seq, Feature)로 받음
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (Batch, 784, 1)
        
        # LSTM 초기 은닉 상태 및 셀 상태 (0으로 초기화)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
        
        # LSTM 순전파
        # out shape: (Batch, Seq_Length, Hidden_Size)
        out, _ = self.lstm(x, (h0, c0))
        
        # 마지막 타임스텝(Time step)의 결과만 가져와서 분류
        # out[:, -1, :] shape: (Batch, Hidden_Size)
        out = self.fc(out[:, -1, :])
        return out

class SimpleGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, config):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.device = config.device
        self.config = config
        # batch_first=True: 입력 형태 (Batch, Seq, Feature)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (Batch, Seq_Length, Input_Size)
        
        # GRU 초기 은닉 상태 설정
        # shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
        
        # GRU 순전파
        # out shape: (Batch, Seq_Length, Hidden_Size)
        out, _ = self.gru(x, h0)
        
        # 마지막 타임스텝(Time step)의 결과만 가져와서 분류
        # out[:, -1, :] shape: (Batch, Hidden_Size)
        out = self.fc(out[:, -1, :])
        return out

class SimpleMamba(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, config):
        super(SimpleMamba, self).__init__()
        self.hidden_size = hidden_size
        self.device = config.device
        self.config = config
        # 입력을 hidden_size로 투영 (Mamba는 입력과 출력 차원이 같아야 함)
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # Mamba 블록 설정
        self.mamba = Mamba(
            d_model=hidden_size,  # 모델 차원
            d_state=16,           # SSM 상태 차원
            d_conv=4,            # 로컬 컨볼루션 커널 크기
            expand=2,            # 확장 계수 (내부적으로 2배 차원을 키워 연산)
        ).to(self.device)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (Batch, Seq_Length, Input_Size) -> (Batch, 784, 1)
        
        # 입력 차원 맞추기 (1 -> hidden_size)
        x = self.embedding(x)
        
        # Mamba 순전파 
        # out shape: (Batch, Seq_Length, Hidden_Size)
        out = self.mamba(x)
        
        # 마지막 타임스텝의 결과 사용 
        # out[:, -1, :] shape: (Batch, Hidden_Size)
        out = self.fc(out[:, -1, :])
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=784):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SimpleTransformer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, config):
        super(SimpleTransformer, self).__init__()
        self.d_model = hidden_size # Transformer에서는 hidden_size를 d_model로 사용
        self.device = config.device
        self.config = config

        # positional encoding
        self.embedding = nn.Linear(input_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # 트랜스포머 인코더 레이어
        # nhead should be divided by d_model
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=4, 
            dim_feedforward=self.d_model * 4, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        
        # 분류기
        self.fc = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        # x shape: (Batch, 784, 1)
        
        # 임베딩 + 위치 정보 추가
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # 트랜스포머 연산
        # out shape: (Batch, 784, d_model)
        out = self.transformer_encoder(x)
        
        # 마지막 타임스텝 혹은 평균 풀링 사용 (여기서는 마지막 시점 사용)
        # out = self.fc(out[:, -1, :])
        out = out.mean(dim=1)
        return out
    
class LearnableDelayRNN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int, max_delay:int, config:Config):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_delay = max_delay
        self.device = config.device
        self.config = config
        # 기본 가중치
        self.afferent = nn.Linear(input_size, hidden_size)
        self.lateral = nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty(hidden_size, hidden_size)) # hidden_out, hidden_in
            )
        self.efferent = nn.Linear(hidden_size, output_size)
        
        self.tau = nn.Parameter(max_delay * torch.rand_like(self.lateral) + 1)
        self.sigma = max_delay / 2
    
    @staticmethod
    @torch.jit.script
    def calc_credit_matrix_jit(tau_clipped:torch.Tensor, max_delay:int, hidden_size:int, sigma:float) -> torch.Tensor:
        credit_matrix = torch.arange(max_delay + 1, out=tau_clipped.new_empty(max_delay + 1)) # (max_delay+1,)
        credit_matrix = credit_matrix[:, None, None].repeat(1, hidden_size, hidden_size) # (max_delay+1, hidden_out, hidden_in)

        inv_sigma = 1 / sigma
        credit_matrix = torch.nn.functional.relu(-abs((credit_matrix - tau_clipped) * inv_sigma ** 2) + inv_sigma) # Credit matrix with Gaussian profile
        
        return credit_matrix
    
    def calc_credit_matrix(self) -> Float[torch.Tensor, "max_delay+1 hidden_out hidden_in"]:
        return LearnableDelayRNN.calc_credit_matrix_jit(
            torch.clamp(self.tau, 1, self.max_delay)[None,...],
            self.max_delay,
            self.hidden_size,
            self.sigma)
    
    @staticmethod
    @torch.jit.script
    def step_jit(x_t:torch.Tensor,
                 credit_matrix:torch.Tensor,
                 buffer:torch.Tensor,
                 buffer_ptr:int,
                 lateral:torch.Tensor,
                 max_delay:int,
                 w_afferent:torch.Tensor,
                 b_afferent:torch.Tensor,
                 w_efferent:torch.Tensor,
                 b_efferent:torch.Tensor):
        h_delayed = buffer[buffer_ptr]  # Get the current delayed hidden state
        h_to_delay = torch.tanh(torch.nn.functional.linear(x_t, w_afferent, b_afferent) + h_delayed) # lateral is processed via buffer, already included in h_delayed
        
        
        ### Update buffer with new hidden state to be delayed
        credit_matrix = credit_matrix * lateral[None, :, :]  # (max_delay+1, hidden_out, hidden_in)
        scattered = torch.einsum('dhi,bi->dbh', credit_matrix, h_to_delay)  # (max_delay+1, batch_size, hidden_out)
        
        # Shift the scattered values according to the current buffer pointer
        shifted_scattered = torch.roll(scattered, shifts=buffer_ptr, dims=0)
        
        # Add the shifted scattered values to the buffer, and update the buffer pointer (removing the oldest value)
        buffer = buffer + shifted_scattered
        mask = torch.arange(max_delay + 1, out=buffer.new_empty(max_delay + 1)) == buffer_ptr # Remove the oldest value
        buffer = buffer * (~mask[:, None, None])  # Zero out the position at buffer_ptr
        buffer_ptr = (buffer_ptr + 1) % (max_delay + 1)  # Update the pointer
        ### End buffer update
        
        
        y_t = torch.nn.functional.linear(h_delayed, w_efferent, b_efferent)
        return buffer, buffer_ptr, y_t # return the next delayed hidden state
    
    def step(self,
             x_t: Float[torch.Tensor, "batch_size input_size"],
             credit_matrix: Float[torch.Tensor, "max_delay+1 hidden_out hidden_in"],
             buffer: Float[torch.Tensor, "max_delay+1 batch_size hidden_size"],
             buffer_ptr: int) \
                 -> tuple[Float[torch.Tensor, "max_delay+1 batch_size hidden_size"], int, Float[torch.Tensor, "batch_size output_size"]]:
        w_afferent, b_afferent = self.afferent.weight, self.afferent.bias
        w_efferent, b_efferent = self.efferent.weight, self.efferent.bias
        return LearnableDelayRNN.step_jit(x_t, credit_matrix, buffer, buffer_ptr, self.lateral, self.max_delay, w_afferent, b_afferent, w_efferent, b_efferent)
    
    def forward(self,
                x: Float[torch.Tensor, "batch_size time input_size"],
                return_seq: torch.Tensor|None = None) \
                    -> Float[torch.Tensor, "batch_size time output_size"]:
        # Precompute credit matrix (Valid until the taus are updated)
        credit_matrix = self.calc_credit_matrix()
        
        # Initialize output tensor
        y = x.new_zeros(*x.shape[:-1], self.output_size)# To store outputs at each time step
        
        # Initialize buffer and get the initial delayed hidden state
        buffer = x.new_zeros(self.max_delay + 1, x.size(0), self.hidden_size) # (max_delay+1, batch_size, hidden_size)
        buffer_ptr = 0  # Pointer to track the current position in the buffer
        
        for t in range(x.size(1)):
            x_t = x[:, t, :] # Batch size, input_size
            buffer, buffer_ptr, y_t = self.step(x_t, credit_matrix, buffer, buffer_ptr)
            y[:, t, :] = y_t
        
        if return_seq is not None:
            if return_seq.shape != y.shape:
                return_seq.resize_(y.shape)
            return_seq.copy_(y)
        
        return y[:, -1, :]  # Return only the last output if return_seq is not specified
    
if __name__ == "__main__":
    # Example usage
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = get_model_with_preset(ModelType.RNN)
    print("SimpleRNN", count_parameters(model), "parameters")

    model = get_model_with_preset(ModelType.LSTM)
    print("SimpleLSTM", count_parameters(model), "parameters")

    model = SimpleGRU(input_size=1, hidden_size=75, num_classes=10, config=config)
    print("SimpleGRU", sum(p.numel() for p in model.parameters() if p.requires_grad), "parameters")
    
    model = SimpleTransformer(input_size=1, hidden_size=32, num_classes=10, config=config)
    print("SimpleTransformer", sum(p.numel() for p in model.parameters() if p.requires_grad), "parameters")
    
    model = get_model_with_preset(ModelType.DelayedRNN)
    print("LearnableDelayRNN", count_parameters(model), "parameters")