from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from utils.config import Config, ModelType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

presets={
    ModelType.RNN: Config(
        model_type=ModelType.RNN,
        max_delay=40,
        max_think_steps=100,
        seed=None,
        batch_size=32,
        input_size=11,
        # seq_length=784,
        seq_min=5,
        seq_max=20,
        hidden_size=256,
        num_classes=10,
        learning_rate=0.01,
        epochs=100),
    ModelType.LSTM: Config(
        model_type=ModelType.LSTM,
        max_delay=40,
        max_think_steps=100,
        seed=None,
        batch_size=32,
        input_size=11,
        # seq_length=784,
        seq_min=5,
        seq_max=20,
        hidden_size=128,
        num_classes=10,
        learning_rate=0.01,
        epochs=100),
    ModelType.DelayedRNN: Config(
        model_type=ModelType.DelayedRNN,
        max_delay=40,
        max_think_steps=100,
        seed=None,
        batch_size=32,
        input_size=11,
        # seq_length=784,
        seq_min=5,
        seq_max=20,
        hidden_size=180,
        num_classes=10,
        learning_rate=0.01,
        epochs=100),
}

def get_model_with_preset(model_class:ModelType) -> nn.Module:
    config = presets[model_class]
    match config.model_type:
        case ModelType.RNN:
            return ThinkingRNN(config.input_size, config.hidden_size, config.num_classes, config).to(config.device)
        case ModelType.LSTM:
            return ThinkingLSTM(config.input_size, config.hidden_size, config.num_classes, config).to(config.device)
        case ModelType.DelayedRNN:
            return ThinkingLearnableDelayRNN(config.input_size, config.hidden_size, config.num_classes, config.max_delay, config).to(config.device)
        case _:
            raise ValueError(f"Unsupported model type: {config.model_type}")


@dataclass
class Seq2SeqOutput:
    outputs: Float[torch.Tensor, "batch_size N vocab_size"]
    # logits: Float[torch.Tensor, "batch_size N vocab_size"]
    think_steps: Int[torch.Tensor, "batch_size"]

class ThinkingRNN(nn.Module):
    def __init__(self, input_size:int, hidden_size: int, num_classes: int, config):
        super(ThinkingRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_think_steps = config.max_think_steps
        self.cfg = config
        
        # 기존 클래스 수에 '생각 종료(Think End)' 토큰을 추가하여 전체 단어장 크기 설정
        self.vocab_size = num_classes + 1 
        self.think_end_token = num_classes  # 인덱스 0 ~ num_classes-1은 일반 클래스, num_classes가 '생각 끝' 토큰
        
        # 원핫 인코딩된 입력을 받으므로 input_size는 vocab_size와 동일
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.vocab_size)
        self.device = config.device

    def forward(self, x, lengths, N=None) -> Seq2SeqOutput:
        """
        추론(Inference) 시퀀스 생성 함수
        x shape: (Batch, N+3, vocab_size) - 원핫 인코딩된 입력 시퀀스
        """
        batch_size = x.size(0)
        
        # N값이 명시되지 않았다면 입력 형태(N+3)에서 N을 유추합니다.
        if N is None:
            N = x.size(1) - 3  
        
        # --- 1. 인코딩 단계 (주어진 N+3 시퀀스 읽기) ---
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        lengths_cpu = lengths.cpu() # pack_padded_sequence는 CPU 텐서를 요구합니다.
        packed_x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed_x, h0)
        
        # [핵심 수정] 패딩을 건너뛰고 각 배치의 '진짜 마지막 토큰'을 디코더의 첫 입력으로 설정
        batch_indices = torch.arange(batch_size, device=self.device)
        dec_input = x[batch_indices, lengths - 1, :].unsqueeze(1)
        
        # 최종 N 길이의 결과를 저장할 빈 텐서
        final_outputs = torch.zeros(batch_size, N, self.vocab_size).to(self.device)
        think_steps_list = []
        
        # 배치 내 각 샘플마다 '생각하는 시간'이 다를 수 있으므로 개별 처리합니다.
        for i in range(batch_size):
            h_i = h[:, i:i+1, :]
            curr_input = dec_input[i:i+1, :, :]
            
            # --- 2. 생각(Thinking) 단계 ---
            think_steps = 0
            while True:
                out, h_i = self.rnn(curr_input, h_i)
                logits = self.fc(out.squeeze(1))  # (1, vocab_size)
                # probs = F.softmax(logits, dim=-1)
                
                # # Softmax 확률 분포에서 1개 샘플링
                # sampled_idx = torch.multinomial(probs, 1)
                
                # # 샘플링된 인덱스를 다시 원핫 인코딩하여 다음 스텝 입력으로 변환
                # curr_input = F.one_hot(sampled_idx, num_classes=self.vocab_size).float().unsqueeze(1)
                curr_input = F.gumbel_softmax(logits, tau=1.0, hard=True).unsqueeze(1)
                
                think_steps += 1
                
                # '생각 끝' 토큰이 뽑히거나, 무한 루프에 빠지는 것을 방지(최대 100번)하면 생각 종료
                if curr_input.argmax(dim=-1).item() == self.think_end_token or think_steps > self.max_think_steps:
                    break  
            think_steps_list.append(think_steps)
            
            # --- 3. 출력(Output) 단계 (정확히 N 길이만큼만 생성) ---
            for j in range(N):
                out, h_i = self.rnn(curr_input, h_i)
                logits = self.fc(out.squeeze(1))
                # probs = F.softmax(logits, dim=-1)
                
                # sampled_idx = torch.multinomial(probs, 1)
                # one_hot_out = F.one_hot(sampled_idx, num_classes=self.vocab_size).float()
                
                one_hot_out = F.gumbel_softmax(logits, tau=1.0, hard=True)
                
                # 최종 출력 텐서에 기록하고 다음 입력으로 사용
                final_outputs[i, j, :] = one_hot_out.squeeze(0)
                curr_input = one_hot_out.unsqueeze(1)
                
        return Seq2SeqOutput(outputs=final_outputs[:,:,:-1], # '생각 끝' 토큰을 제외한 실제 클래스 확률만 반환
                             think_steps=torch.tensor(think_steps_list, device=self.device))

class ThinkingLSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size: int, num_classes: int, config):
        super(ThinkingLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_think_steps = config.max_think_steps
        self.cfg = config
        
        # 단어장 크기 (클래스 수 + '생각 끝' 토큰)
        self.vocab_size = num_classes + 1
        self.think_end_token = num_classes  
        self.device = config.device
        
        # LSTM으로 변경: 원핫 입력을 받으므로 input_size = vocab_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, x, lengths, N=None) -> Seq2SeqOutput:
        batch_size = x.size(0)
        
        if N is None:
            N = x.size(1) - 3  
        
        # --- 1. 인코딩 단계 ---
        # LSTM은 h0(단기 기억)와 c0(장기 기억) 두 가지 상태를 가집니다.
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        
        # [핵심 수정] 패딩 오염 방지용 pack_padded_sequence 적용
        lengths_cpu = lengths.cpu()
        packed_x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (h, c) = self.lstm(packed_x, (h0, c0))
        _, (h, c) = self.lstm(x, (h0, c0))
        
        # [핵심 수정] 진짜 마지막 토큰 인덱싱
        batch_indices = torch.arange(batch_size, device=self.device)
        dec_input = x[batch_indices, lengths - 1, :].unsqueeze(1)
        
        # 최종 결과를 저장할 빈 텐서 (Loss 계산을 위해 Logits 형태 유지)
        final_outputs = torch.zeros(batch_size, N, self.vocab_size).to(self.device)
        think_steps_list = []
        
        for i in range(batch_size):
            # LSTM의 은닉 상태와 셀 상태를 각각 슬라이싱
            h_i = h[:, i:i+1, :]
            c_i = c[:, i:i+1, :]
            curr_input = dec_input[i:i+1, :, :]
            
            # --- 2. 생각(Thinking) 단계 ---
            think_steps = 0
            while True:
                # 상태 튜플 (h_i, c_i)를 전달하고 업데이트
                out, (h_i, c_i) = self.lstm(curr_input, (h_i, c_i))
                logits = self.fc(out.squeeze(1))
                
                # 미분 가능한 샘플링 (Gumbel-Softmax)
                # hard=True로 설정하여 겉모양은 원핫 벡터지만, 내부는 Softmax 기울기를 가짐
                curr_input_squeeze = F.gumbel_softmax(logits, tau=1.0, hard=True)
                curr_input = curr_input_squeeze.unsqueeze(1)
                
                sampled_idx = curr_input_squeeze.argmax(dim=-1).item()
                think_steps += 1
                
                if sampled_idx == self.think_end_token or think_steps > self.max_think_steps:
                    break  
            think_steps_list.append(think_steps)
            
            # --- 3. 출력(Output) 단계 ---
            for j in range(N):
                out, (h_i, c_i) = self.lstm(curr_input, (h_i, c_i))
                logits = self.fc(out.squeeze(1))
                
                # [중요] CrossEntropyLoss에 그대로 넣을 수 있도록 Logits(정답) 자체를 저장
                final_outputs[i, j, :] = logits
                
                # 다음 타임스텝의 입력은 Gumbel-Softmax를 거쳐 원핫 형태로 변환하여 전달
                one_hot_out = F.gumbel_softmax(logits, tau=1.0, hard=True)
                curr_input = one_hot_out.unsqueeze(1)
                
        return Seq2SeqOutput(outputs=final_outputs[:,:,:-1], think_steps=torch.tensor(think_steps_list, device=self.device))

class ThinkingLearnableDelayRNN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_classes:int, max_delay:int, config:Config):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_think_steps = config.max_think_steps
        self.cfg = config
        self.max_delay = max_delay
        self.device = config.device
        
        # 단어장 크기 설정 (기존 클래스 + 생각 끝 토큰)
        self.vocab_size = num_classes + 1
        self.think_end_token = num_classes
        
        # 입출력 크기를 vocab_size로 통일
        self.output_size = self.vocab_size
        
        # 기본 가중치
        self.afferent = nn.Linear(self.input_size, hidden_size)
        self.lateral = nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty(hidden_size, hidden_size)) # hidden_out, hidden_in
            )
        self.efferent = nn.Linear(hidden_size, self.output_size)
        
        self.tau = nn.Parameter(max_delay * torch.rand_like(self.lateral) + 1)
        self.sigma = max_delay / 2

    @staticmethod
    @torch.jit.script
    def calc_credit_matrix_jit(tau_clipped:torch.Tensor, max_delay:int, hidden_size:int, sigma:float) -> torch.Tensor:
        # (기존 코드와 동일)
        credit_matrix = torch.arange(max_delay + 1, out=tau_clipped.new_empty(max_delay + 1)) 
        credit_matrix = credit_matrix[:, None, None].repeat(1, hidden_size, hidden_size) 

        inv_sigma = 1 / sigma
        credit_matrix = torch.nn.functional.relu(-abs((credit_matrix - tau_clipped) * inv_sigma ** 2) + inv_sigma) 
        
        return credit_matrix
    
    def calc_credit_matrix(self): # -> Float[torch.Tensor, "max_delay+1 hidden_out hidden_in"]:
        return ThinkingLearnableDelayRNN.calc_credit_matrix_jit(
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
        # (기존 코드와 동일 - JIT 컴파일 유지)
        h_delayed = buffer[buffer_ptr]  
        h_to_delay = torch.tanh(torch.nn.functional.linear(x_t, w_afferent, b_afferent) + h_delayed) 
        
        credit_matrix = credit_matrix * lateral[None, :, :]  
        scattered = torch.einsum('dhi,bi->dbh', credit_matrix, h_to_delay)  
        
        shifted_scattered = torch.roll(scattered, shifts=buffer_ptr, dims=0)
        
        buffer = buffer + shifted_scattered
        mask = torch.arange(max_delay + 1, out=buffer.new_empty(max_delay + 1)) == buffer_ptr 
        buffer = buffer * (~mask[:, None, None])  
        buffer_ptr = (buffer_ptr + 1) % (max_delay + 1)  
        
        y_t = torch.nn.functional.linear(h_delayed, w_efferent, b_efferent)
        return buffer, buffer_ptr, y_t 
    
    def step(self, x_t, credit_matrix, buffer, buffer_ptr):
        w_afferent, b_afferent = self.afferent.weight, self.afferent.bias
        w_efferent, b_efferent = self.efferent.weight, self.efferent.bias
        return ThinkingLearnableDelayRNN.step_jit(x_t, credit_matrix, buffer, buffer_ptr, self.lateral, self.max_delay, w_afferent, b_afferent, w_efferent, b_efferent)
    
    def forward(self,
                x:Float[torch.Tensor, "batch_size time_steps input_size"],
                lengths:Int[torch.Tensor, "batch_size"],
                N=None) -> Seq2SeqOutput:
        """
        x shape: (batch_size, time_steps, input_size)
        lengths shape: (batch_size,)
        """
        batch_size = x.size(0)
        if N is None:
            N = x.size(1) - 3  # N 유추

        credit_matrix = self.calc_credit_matrix()
        
        # 1. 초기화 및 인코딩(Encoding) 단계
        # 전체 배치에 대한 buffer 초기화
        buffer = x.new_zeros(self.max_delay + 1, batch_size, self.hidden_size)
        buffer_ptr = 0 
        
        # [핵심 로직 1] 패딩에 오염되기 전의 상태를 저장할 빈 텐서 준비
        saved_buffers = buffer.new_zeros(buffer.size())
        saved_buffer_ptrs = buffer.new_zeros(batch_size, dtype=torch.long)
        
        # 입력 시퀀스를 순차적으로 읽으며 buffer에 문맥 축적
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            buffer, buffer_ptr, _ = self.step(x_t, credit_matrix, buffer, buffer_ptr)
            
            # [핵심 로직 2] 방금 계산한 t가 어떤 배치의 '진짜 마지막 단어'인지 확인
            is_last_token:torch.Tensor = (t == lengths - 1) # type: ignore # shape: (batch_size,)
            
            if is_last_token.any():
                # 문장이 끝난 배치들만 현재의 깨끗한 buffer와 포인터를 복사해 둡니다.
                mask = is_last_token.view(1, batch_size, 1) # broadcasting을 위해 형태 변경
                saved_buffers = torch.where(mask, buffer, saved_buffers)
                saved_buffer_ptrs[is_last_token] = buffer_ptr
            
        # 디코더의 첫 입력 (진짜 마지막 토큰 추출)
        batch_indices = torch.arange(batch_size, out=x.new_empty(batch_size, dtype=torch.long))
        dec_input = x[batch_indices, lengths - 1, :]
        
        # 최종 결과를 저장할 텐서 (Logits 유지)
        final_outputs = x.new_zeros(batch_size, N, self.vocab_size)
        think_steps_list = []
        
        # 배치별로 생각하는(Thinking) 시간이 다르므로 독립적으로 분리하여 연산
        for i in range(batch_size):
            # [핵심 로직 3] 루프를 끝까지 돌아 패딩에 오염된 원래 buffer 대신, 
            # 아까 저장해둔 깨끗한 saved_buffers에서 상태를 불러와 생각을 시작합니다.
            buffer_i = saved_buffers[:, i:i+1, :]     
            buffer_ptr_i = saved_buffer_ptrs[i].item() # 저장된 정수형 포인터
            curr_input = dec_input[i:i+1, :]
            
            # --- 2. 생각(Thinking) 단계 ---
            think_steps = 0
            while True:
                # step 함수 호출
                buffer_i, buffer_ptr_i, y_t = self.step(curr_input, credit_matrix, buffer_i, buffer_ptr_i)
                
                # Gumbel-Softmax 샘플링 (미분 가능)
                curr_input = F.gumbel_softmax(y_t, tau=1.0, hard=True)
                
                sampled_idx = curr_input.argmax(dim=-1).item()
                think_steps += 1
                
                if sampled_idx == self.think_end_token or think_steps > self.max_think_steps:
                    break  
            think_steps_list.append(think_steps)
            
            # --- 3. 출력(Output) 단계 ---
            for j in range(N):
                buffer_i, buffer_ptr_i, y_t = self.step(curr_input, credit_matrix, buffer_i, buffer_ptr_i)
                
                # 정답 계산(Loss)을 위해 Logits 자체를 저장
                final_outputs[i, j, :] = y_t.squeeze(0)
                
                # 다음 스텝 입력을 위한 샘플링
                curr_input = F.gumbel_softmax(y_t, tau=1.0, hard=True)
                
        return Seq2SeqOutput(outputs=final_outputs[:,:,:-1], think_steps=torch.tensor(think_steps_list, device=self.device))
    
if __name__ == "__main__":
    # Example usage
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = get_model_with_preset(ModelType.RNN)
    print("SimpleRNN", count_parameters(model), "parameters")

    model = get_model_with_preset(ModelType.LSTM)
    print("SimpleLSTM", count_parameters(model), "parameters")

    # model = get_model_with_preset(ModelType.GRU)
    # print("SimpleGRU", count_parameters(model), "parameters")
    
    model = get_model_with_preset(ModelType.DelayedRNN)
    print("LearnableDelayRNN", count_parameters(model), "parameters")