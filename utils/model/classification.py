from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float

from utils.config import Config, ModelType

presets={
    ModelType.RNN: Config(model_type=ModelType.RNN),
    ModelType.LSTM: Config(model_type=ModelType.LSTM),
    ModelType.DelayedRNN: Config(model_type=ModelType.DelayedRNN)
}

def get_model_with_preset(model_class:ModelType) -> nn.Module:
    config = presets[model_class]
    match config.model_type:
        case ModelType.RNN:
            return SimpleRNN(config.input_size, config.hidden_size, config.num_classes, config).to(config.device)
        case ModelType.LSTM:
            return SimpleLSTM(config.input_size, config.hidden_size, config.num_classes, config).to(config.device)
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

class LearnableDelayRNN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int, max_delay:int, config:Config):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_delay = max_delay
        self.device = config.device
        
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
    config = Config(device='cpu')
    model = SimpleRNN(input_size=1, hidden_size=128, num_classes=10, config=config)
    print("SimpleRNN", sum(p.numel() for p in model.parameters() if p.requires_grad), "parameters")
    model = SimpleLSTM(input_size=1, hidden_size=64, num_classes=10, config=config)
    print("SimpleLSTM", sum(p.numel() for p in model.parameters() if p.requires_grad), "parameters")
    model = LearnableDelayRNN(input_size=1, hidden_size=64, output_size=10, max_delay=20, config=config)
    print("LearnableDelayRNN", sum(p.numel() for p in model.parameters() if p.requires_grad), "parameters")