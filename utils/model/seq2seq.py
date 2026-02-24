from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float

from utils.config import Config

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x, N=None):
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
        _, h = self.rnn(x, h0)  # 마지막 은닉 상태 h를 전체 문맥(Context)으로 사용
        
        # 디코더의 첫 입력은 입력 시퀀스의 마지막 토큰으로 시작
        dec_input = x[:, -1, :].unsqueeze(1)  # shape: (Batch, 1, vocab_size)
        
        # 최종 N 길이의 결과를 저장할 빈 텐서
        final_outputs = torch.zeros(batch_size, N, self.vocab_size).to(self.device)
        
        # 배치 내 각 샘플마다 '생각하는 시간'이 다를 수 있으므로 개별 처리합니다.
        for i in range(batch_size):
            h_i = h[:, i:i+1, :]
            curr_input = dec_input[i:i+1, :, :]
            
            # --- 2. 생각(Thinking) 단계 ---
            think_steps = 0
            while True:
                out, h_i = self.rnn(curr_input, h_i)
                logits = self.fc(out.squeeze(1))  # (1, vocab_size)
                probs = F.softmax(logits, dim=-1)
                
                # # Softmax 확률 분포에서 1개 샘플링
                # sampled_idx = torch.multinomial(probs, 1)
                
                # # 샘플링된 인덱스를 다시 원핫 인코딩하여 다음 스텝 입력으로 변환
                # curr_input = F.one_hot(sampled_idx, num_classes=self.vocab_size).float().unsqueeze(1)
                curr_input = F.gumbel_softmax(logits, tau=1.0, hard=True).unsqueeze(1)
                
                think_steps += 1
                
                # '생각 끝' 토큰이 뽑히거나, 무한 루프에 빠지는 것을 방지(최대 100번)하면 생각 종료
                if curr_input.argmax(dim=-1).item() == self.think_end_token or think_steps > self.max_think_steps:
                    break  
            
            # --- 3. 출력(Output) 단계 (정확히 N 길이만큼만 생성) ---
            for j in range(N):
                out, h_i = self.rnn(curr_input, h_i)
                logits = self.fc(out.squeeze(1))
                probs = F.softmax(logits, dim=-1)
                
                # sampled_idx = torch.multinomial(probs, 1)
                # one_hot_out = F.one_hot(sampled_idx, num_classes=self.vocab_size).float()
                
                one_hot_out = F.gumbel_softmax(logits, tau=1.0, hard=True)
                
                # 최종 출력 텐서에 기록하고 다음 입력으로 사용
                final_outputs[i, j, :] = one_hot_out.squeeze(0)
                curr_input = one_hot_out.unsqueeze(1)
                
        return final_outputs[:,:,:-1]  # '생각 끝' 토큰을 제외한 실제 클래스 확률만 반환

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x, N=None):
        batch_size = x.size(0)
        
        if N is None:
            N = x.size(1) - 3  
        
        # --- 1. 인코딩 단계 ---
        # LSTM은 h0(단기 기억)와 c0(장기 기억) 두 가지 상태를 가집니다.
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        
        # 전체 시퀀스 문맥 파악
        _, (h, c) = self.lstm(x, (h0, c0))
        
        # 디코더의 첫 입력
        dec_input = x[:, -1, :].unsqueeze(1)  
        
        # 최종 결과를 저장할 빈 텐서 (Loss 계산을 위해 Logits 형태 유지)
        final_outputs = torch.zeros(batch_size, N, self.vocab_size).to(self.device)
        
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
            
            # --- 3. 출력(Output) 단계 ---
            for j in range(N):
                out, (h_i, c_i) = self.lstm(curr_input, (h_i, c_i))
                logits = self.fc(out.squeeze(1))
                
                # [중요] CrossEntropyLoss에 그대로 넣을 수 있도록 Logits(정답) 자체를 저장
                final_outputs[i, j, :] = logits
                
                # 다음 타임스텝의 입력은 Gumbel-Softmax를 거쳐 원핫 형태로 변환하여 전달
                one_hot_out = F.gumbel_softmax(logits, tau=1.0, hard=True)
                curr_input = one_hot_out.unsqueeze(1)
                
        return final_outputs[:,:,:-1]
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# 타입 힌트를 위한 임시 정의 (실제 환경에 맞게 조정하세요)
# from jaxtyping import Float

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
    
    def forward(self, x, N=None):
        """
        x shape: (batch_size, time_steps, input_size)
        """
        batch_size = x.size(0)
        if N is None:
            N = x.size(1) - 3  # N 유추

        credit_matrix = self.calc_credit_matrix()
        
        # 1. 초기화 및 인코딩(Encoding) 단계
        # 전체 배치에 대한 buffer 초기화
        buffer = x.new_zeros(self.max_delay + 1, batch_size, self.hidden_size)
        buffer_ptr = 0  
        
        # 입력 시퀀스를 순차적으로 읽으며 buffer에 문맥 축적
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            buffer, buffer_ptr, _ = self.step(x_t, credit_matrix, buffer, buffer_ptr)
            
        # 디코더의 첫 입력 (입력 시퀀스의 마지막 토큰)
        dec_input = x[:, -1, :] # shape: (batch_size, vocab_size)
        
        # 최종 결과를 저장할 텐서 (Logits 유지)
        final_outputs = torch.zeros(batch_size, N, self.vocab_size).to(self.device)
        
        # 배치별로 생각하는(Thinking) 시간이 다르므로 독립적으로 분리하여 연산
        for i in range(batch_size):
            # i번째 배치의 상태만 슬라이싱 (형태를 유지하기 위해 i:i+1 사용)
            buffer_i = buffer[:, i:i+1, :]     # (max_delay+1, 1, hidden_size)
            buffer_ptr_i = buffer_ptr          # 포인터는 정수이므로 복사
            curr_input = dec_input[i:i+1, :]   # (1, vocab_size)
            
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
            
            # --- 3. 출력(Output) 단계 ---
            for j in range(N):
                buffer_i, buffer_ptr_i, y_t = self.step(curr_input, credit_matrix, buffer_i, buffer_ptr_i)
                
                # 정답 계산(Loss)을 위해 Logits 자체를 저장
                final_outputs[i, j, :] = y_t.squeeze(0)
                
                # 다음 스텝 입력을 위한 샘플링
                curr_input = F.gumbel_softmax(y_t, tau=1.0, hard=True)
                
        return final_outputs[:,:,:-1]  # '생각 끝' 토큰 제외한 실제 클래스 확률 반환