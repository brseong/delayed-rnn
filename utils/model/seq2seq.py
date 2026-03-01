from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from math import log
import torch
import torch.nn as nn
from jaxtyping import Float, Int

from utils.config import Config, ModelType
from utils.random import clipped_gamma_sample

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus
from torch.nn.utils.rnn import pack_padded_sequence


presets={
    ModelType.RNN: Config(
        model_type=ModelType.RNN,
        # max_delay=40,
        max_think_steps=100,
        seed=None,
        batch_size=32,
        input_size=11,
        # seq_length=784,
        seq_min=5,
        seq_max=20,
        hidden_size=512,
        num_classes=10,
        learning_rate=0.01,
        epochs=100),
    ModelType.LSTM: Config(
        model_type=ModelType.LSTM,
        # max_delay=40,
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
    ModelType.GRU: Config(
        model_type=ModelType.GRU,
        # max_delay=40,
        max_think_steps=100,
        seed=None,
        batch_size=32,
        input_size=11,
        # seq_length=784,
        seq_min=5,
        seq_max=20,
        hidden_size=296,
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
        hidden_size=360,
        num_classes=10,
        learning_rate=0.01,
        epochs=100),
}

def get_model(model_class:ModelType, device:torch.device, config:Config|None=None) -> ThinkingRNN | ThinkingLSTM | FastThinkingLearnableDelayRNN | ThinkingGRU:
    if config is None:
        config = presets[model_class]
        config.device = device  # Set the device in the config for later use in model initialization
    match config.model_type:
        case ModelType.RNN:
            return ThinkingRNN(config.input_size, config.hidden_size, config.num_classes, config=config).to(config.device)
        case ModelType.LSTM:
            return ThinkingLSTM(config.input_size, config.hidden_size, config.num_classes, config=config).to(config.device)
        case ModelType.DelayedRNN:
            # return ThinkingLearnableDelayRNN(config.input_size, config.hidden_size, config.num_classes, max_delay=config.max_delay, config=config).to(config.device)
            return FastThinkingLearnableDelayRNN(config.input_size, config.hidden_size, config.num_classes, max_delay=config.max_delay, config=config).to(config.device)
        case ModelType.GRU:
            return ThinkingGRU(config.input_size, config.hidden_size, config.num_classes, config=config).to(config.device)
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
        self.config = config
        
        # 기존 클래스 수에 '생각 종료(Think End)' 토큰을 추가하여 전체 단어장 크기 설정
        self.vocab_size = num_classes + 1 
        self.think_end_token = num_classes  # 인덱스 0 ~ num_classes-1은 일반 클래스, num_classes가 '생각 끝' 토큰
        
        # 원핫 인코딩된 입력을 받으므로 input_size는 vocab_size와 동일
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.vocab_size)
        self.device = config.device

    def forward(self, x, lengths, N=None, targets=None, teacher_forcing_ratio=0.0) -> Seq2SeqOutput:
        """
        추론(Inference) 시퀀스 생성 함수 (배치 처리 최적화 완료)
        x shape: (Batch, N+3, vocab_size) - 원핫 인코딩된 입력 시퀀스
        """
        batch_size = x.size(0)
        
        if N is None:
            N = x.size(1) - 3  
        
        # --- 1. 인코딩 단계 (주어진 N+3 시퀀스 읽기) ---
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        lengths_cpu = lengths.cpu() 
        packed_x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed_x, h0)
        
        # 패딩을 건너뛰고 각 배치의 '진짜 마지막 토큰'을 디코더의 첫 입력으로 설정
        batch_indices = torch.arange(batch_size, device=self.device)
        dec_input = x[batch_indices, lengths - 1, :].unsqueeze(1)
        
        # 최종 N 길이의 결과를 저장할 빈 텐서
        final_outputs = torch.zeros(batch_size, N, self.vocab_size, device=self.device)
        
        # 배치 내 각 샘플의 상태를 추적하기 위한 텐서들
        think_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        output_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        is_thinking = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        curr_input = dec_input
        
        # --- 2 & 3. 생각(Thinking) & 출력(Output) 동시 배치 처리 ---
        while True:
            # 모든 배치가 목표 출력 길이(N)를 달성하면 완전 종료
            if (output_steps == N).all():
                break
                
            # [핵심] 현재 스텝에서 '출력 단계'인 배치 확인 (생각이 이미 끝났고, 아직 N개를 다 못 채운 경우)
            outputting_mask = (~is_thinking) & (output_steps < N)
            
            # 현재 스텝에서 '여전히 생각 중'인 배치 확인
            thinking_mask = is_thinking.clone()
            
            # RNN Forward (전체 배치 한 번에 통과)
            out, h = self.rnn(curr_input, h)
            logits = self.fc(out.squeeze(1))  # (batch_size, vocab_size)
            one_hot_out = F.gumbel_softmax(logits, tau=1.0, hard=True) # (batch_size, vocab_size)
            
            # --- 출력(Output) 처리 ---
            out_idx = torch.nonzero(outputting_mask, as_tuple=True)[0]
            if out_idx.numel() > 0:
                step_idx = output_steps[out_idx]
                
                # 예측된 토큰을 최종 출력 텐서에 기록
                final_outputs[out_idx, step_idx, :] = one_hot_out[out_idx]
                
                # Teacher Forcing 적용 (None 체크 오류 수정됨)
                if targets is not None:
                    true_tokens = targets[out_idx, step_idx]
                    
                    # -1 (패딩)이 아닌 경우에만 Teacher Forcing 적용 여부 결정
                    tf_rand = torch.rand(out_idx.numel(), device=self.device)
                    do_tf = (tf_rand < teacher_forcing_ratio) & (true_tokens >= 0)
                    
                    if do_tf.any():
                        tf_indices = out_idx[do_tf]
                        tf_step_idx = step_idx[do_tf]
                        tf_true_tokens = targets[tf_indices, tf_step_idx]
                        
                        # Teacher Forcing이 적용된 배치는 다음 입력으로 정답 토큰을 사용하도록 덮어쓰기
                        tf_one_hot = F.one_hot(tf_true_tokens, num_classes=self.vocab_size).float()
                        one_hot_out[tf_indices] = tf_one_hot
                
                # 출력 카운트 증가
                output_steps[outputting_mask] += 1
                
            # --- 생각(Thinking) 처리 ---
            if thinking_mask.any():
                think_steps[thinking_mask] += 1
                
                pred_tokens = one_hot_out.argmax(dim=-1)
                hit_end = (pred_tokens == self.think_end_token)
                hit_max = (think_steps >= self.max_think_steps)
                
                # '생각 끝' 토큰이 뽑혔거나 최대 생각 횟수를 초과한 경우 해당 배치의 생각 종료
                just_finished = thinking_mask & (hit_end | hit_max)
                is_thinking[just_finished] = False
                
            # 다음 스텝의 입력 준비 (Teacher Forcing이 적용되었다면 정답 토큰이 들어감)
            curr_input = one_hot_out.unsqueeze(1)
            
        return Seq2SeqOutput(
            outputs=final_outputs[:, :, :-1], # '생각 끝' 토큰을 제외한 실제 클래스 확률만 반환
            think_steps=think_steps
        )

class ThinkingLSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size: int, num_classes: int, config):
        super(ThinkingLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_think_steps = config.max_think_steps
        self.config = config
        
        # 단어장 크기 (클래스 수 + '생각 끝' 토큰)
        self.vocab_size = num_classes + 1
        self.think_end_token = num_classes  
        self.device = config.device
        
        # LSTM으로 변경: 원핫 입력을 받으므로 input_size = vocab_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.vocab_size)
    
    def forward(self, x, lengths, N=None, targets=None, teacher_forcing_ratio=0.0) -> Seq2SeqOutput:
        batch_size = x.size(0)
        
        if N is None:
            N = x.size(1) - 3  
        
        # --- 1. 인코딩 단계 ---
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        
        lengths_cpu = lengths.cpu()
        packed_x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (h, c) = self.lstm(packed_x, (h0, c0))
        # [수정됨] 기존 코드에 있던 _, (h, c) = self.lstm(x, (h0, c0)) 는 패딩 오염을 유발하므로 삭제했습니다.
        
        batch_indices = torch.arange(batch_size, device=self.device)
        dec_input = x[batch_indices, lengths - 1, :].unsqueeze(1)
        
        # 최종 결과를 저장할 빈 텐서 (Loss 계산을 위해 Logits 형태 유지)
        final_outputs = torch.zeros(batch_size, N, self.vocab_size, device=self.device)
        
        # 상태 추적용 텐서
        think_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        output_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        is_thinking = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        curr_input = dec_input
        
        # --- 2 & 3. 생각(Thinking) & 출력(Output) 동시 배치 처리 ---
        while True:
            if (output_steps == N).all():
                break
                
            outputting_mask = (~is_thinking) & (output_steps < N)
            thinking_mask = is_thinking.clone()
            
            # LSTM Forward (h와 c를 튜플로 전달)
            out, (h, c) = self.lstm(curr_input, (h, c))
            logits = self.fc(out.squeeze(1))
            
            # Gumbel-Softmax로 다음 입력 후보 생성
            one_hot_out = F.gumbel_softmax(logits, tau=1.0, hard=True)
            
            # --- 출력(Output) 처리 ---
            out_idx = torch.nonzero(outputting_mask, as_tuple=True)[0]
            if out_idx.numel() > 0:
                step_idx = output_steps[out_idx]
                
                # [중요] CrossEntropyLoss를 위해 Logits 자체를 저장
                final_outputs[out_idx, step_idx, :] = logits[out_idx]
                
                # Teacher Forcing
                if targets is not None:
                    true_tokens = targets[out_idx, step_idx]
                    tf_rand = torch.rand(out_idx.numel(), device=self.device)
                    do_tf = (tf_rand < teacher_forcing_ratio) & (true_tokens >= 0)
                    
                    if do_tf.any():
                        tf_indices = out_idx[do_tf]
                        tf_step_idx = step_idx[do_tf]
                        tf_true_tokens = targets[tf_indices, tf_step_idx]
                        
                        tf_one_hot = F.one_hot(tf_true_tokens, num_classes=self.vocab_size).float()
                        one_hot_out[tf_indices] = tf_one_hot
                
                output_steps[outputting_mask] += 1
                
            # --- 생각(Thinking) 처리 ---
            if thinking_mask.any():
                think_steps[thinking_mask] += 1
                
                pred_tokens = one_hot_out.argmax(dim=-1)
                hit_end = (pred_tokens == self.think_end_token)
                hit_max = (think_steps >= self.max_think_steps)
                
                just_finished = thinking_mask & (hit_end | hit_max)
                is_thinking[just_finished] = False
                
            curr_input = one_hot_out.unsqueeze(1)
            
        return Seq2SeqOutput(
            outputs=final_outputs[:, :, :-1], 
            think_steps=think_steps
        )

class ThinkingGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, config):
        super(ThinkingGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_think_steps = config.max_think_steps
        self.config = config
        
        # 단어장 크기 (클래스 수 + '생각 끝' 토큰)
        self.vocab_size = num_classes + 1
        self.think_end_token = num_classes  
        self.device = config.device
        
        # 내부 차원 충돌 방지를 위해 input_size를 vocab_size로 통일
        self.gru = nn.GRU(input_size=self.vocab_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, x, lengths, N=None, targets=None, teacher_forcing_ratio=0.0) -> Seq2SeqOutput:
        batch_size = x.size(0)
        
        if N is None:
            N = x.size(1) - 3  
        
        # --- 1. 인코딩 단계 ---
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        
        lengths_cpu = lengths.cpu()
        packed_x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed_x, h0)
        
        batch_indices = torch.arange(batch_size, device=self.device)
        dec_input = x[batch_indices, lengths - 1, :].unsqueeze(1)  
        
        # 최종 결과를 저장할 빈 텐서 (Logits 형태 유지)
        final_outputs = torch.zeros(batch_size, N, self.vocab_size, device=self.device)
        
        # 상태 추적용 텐서
        think_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        output_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        is_thinking = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        curr_input = dec_input
        
        # --- 2 & 3. 생각(Thinking) & 출력(Output) 동시 배치 처리 ---
        while True:
            if (output_steps == N).all():
                break
                
            outputting_mask = (~is_thinking) & (output_steps < N)
            thinking_mask = is_thinking.clone()
            
            # GRU Forward (h 하나만 주고받음)
            out, h = self.gru(curr_input, h)
            logits = self.fc(out.squeeze(1))
            
            # Gumbel-Softmax로 다음 입력 후보 생성
            one_hot_out = F.gumbel_softmax(logits, tau=1.0, hard=True)
            
            # --- 출력(Output) 처리 ---
            out_idx = torch.nonzero(outputting_mask, as_tuple=True)[0]
            if out_idx.numel() > 0:
                step_idx = output_steps[out_idx]
                
                # Loss 계산을 위해 Logits 자체를 저장
                final_outputs[out_idx, step_idx, :] = logits[out_idx]
                
                # Teacher Forcing
                if targets is not None:
                    true_tokens = targets[out_idx, step_idx]
                    tf_rand = torch.rand(out_idx.numel(), device=self.device)
                    do_tf = (tf_rand < teacher_forcing_ratio) & (true_tokens >= 0)
                    
                    if do_tf.any():
                        tf_indices = out_idx[do_tf]
                        tf_step_idx = step_idx[do_tf]
                        tf_true_tokens = targets[tf_indices, tf_step_idx]
                        
                        tf_one_hot = F.one_hot(tf_true_tokens, num_classes=self.vocab_size).float()
                        one_hot_out[tf_indices] = tf_one_hot
                
                output_steps[outputting_mask] += 1
                
            # --- 생각(Thinking) 처리 ---
            if thinking_mask.any():
                think_steps[thinking_mask] += 1
                
                pred_tokens = one_hot_out.argmax(dim=-1)
                hit_end = (pred_tokens == self.think_end_token)
                hit_max = (think_steps >= self.max_think_steps)
                
                just_finished = thinking_mask & (hit_end | hit_max)
                is_thinking[just_finished] = False
                
            curr_input = one_hot_out.unsqueeze(1)
            
        return Seq2SeqOutput(
            outputs=final_outputs[:, :, :-1], 
            think_steps=think_steps
        )
        
# class ThinkingLearnableDelayRNN(nn.Module):
#     def __init__(self, input_size:int, hidden_size:int, num_classes:int, max_delay:int, config:Config):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.max_think_steps = config.max_think_steps
#         self.config = config
#         self.max_delay = max_delay
#         self.device = config.device
        
#         # 단어장 크기 설정 (기존 클래스 + 생각 끝 토큰)
#         self.vocab_size = num_classes + 1
#         self.think_end_token = num_classes
        
#         # 입출력 크기를 vocab_size로 통일
#         self.output_size = self.vocab_size
        
#         # 기본 가중치
#         self.afferent = nn.Linear(self.input_size, hidden_size)
#         self.lateral = nn.Parameter(
#             torch.nn.init.xavier_uniform_(
#                 torch.empty(hidden_size, hidden_size)) # hidden_out, hidden_in
#             )
#         self.efferent = nn.Linear(hidden_size, self.output_size)
        
#         # time_horizon = 128
#         # tau = torch.empty_like(self.lateral).uniform_(0, log(time_horizon)).clamp_(0, log(max_delay))
#         # self.tau = nn.Parameter(tau)
#         self.tau = nn.Parameter(max_delay * torch.rand_like(self.lateral) + 1)
#         # self.sigma = max_delay / 2
#         self.scale_exponent = nn.Parameter(torch.zeros(hidden_size))

#     @staticmethod
#     @torch.jit.script
#     def calc_credit_matrix_jit(tau_clipped:torch.Tensor, max_delay:int, hidden_size:int, scale_exponent:torch.Tensor) -> torch.Tensor:
#         # (기존 코드와 동일)
#         credit_matrix = torch.arange(max_delay + 1, device=tau_clipped.device).float()  # shape: (max_delay+1,)
#         credit_matrix = credit_matrix[:, None, None]
#         distance = 1.0 + torch.abs(credit_matrix - tau_clipped) # shape: (max_delay+1, hidden_size, hidden_size)

#         # inv_sigma = 1 / sigma
#         # credit_matrix = torch.nn.functional.relu(-abs((credit_matrix - tau_clipped) * inv_sigma ** 2) + inv_sigma)
#         # raw_credit = distance.rsqrt() / distance # Equivalent to distance^(-1.5) but more stable, faster.
        
#         scale_exponent = scale_exponent[None, :, None].sigmoid()  # shape: (1, hidden_size, 1)
#         # raw_credit = distance.reciprocal() # Equivalent to distance^(-1) but more stable, faster.
#         raw_credit = distance.pow(-scale_exponent)
#         credit_matrix = raw_credit / (raw_credit.sum(dim=0, keepdim=True)) # Normalize so that sum of credits across delays equals 1 for each hidden unit
        
#         return credit_matrix
    
#     def calc_credit_matrix(self): # -> Float[torch.Tensor, "max_delay+1 hidden_out hidden_in"]:
#         return ThinkingLearnableDelayRNN.calc_credit_matrix_jit(
#             torch.clamp(self.tau, 1, self.max_delay)[None,...],
#             self.max_delay,
#             self.hidden_size,
#             self.scale_exponent)
    
#     @staticmethod
#     @torch.jit.script
#     def step_jit(x_t:torch.Tensor,
#                  credit_matrix:torch.Tensor,
#                  buffer:torch.Tensor,
#                  buffer_ptr:int,
#                  lateral:torch.Tensor,
#                  max_delay:int,
#                  w_afferent:torch.Tensor,
#                  b_afferent:torch.Tensor,
#                  w_efferent:torch.Tensor,
#                  b_efferent:torch.Tensor):
#         # (기존 코드와 동일 - JIT 컴파일 유지)
#         h_delayed = buffer[buffer_ptr]  
#         h_to_delay = torch.tanh(torch.nn.functional.linear(x_t, w_afferent, b_afferent) + h_delayed) 
        
#         credit_matrix = credit_matrix * lateral[None, :, :]  
#         scattered = torch.einsum('dhi,bi->dbh', credit_matrix, h_to_delay)  
        
#         shifted_scattered = torch.roll(scattered, shifts=buffer_ptr, dims=0)
        
#         buffer = buffer + shifted_scattered
#         mask = torch.arange(max_delay + 1, device=buffer.device) == buffer_ptr 
#         buffer = buffer * (~mask[:, None, None])  
#         buffer_ptr = (buffer_ptr + 1) % (max_delay + 1)  
        
#         y_t = torch.nn.functional.linear(h_delayed, w_efferent, b_efferent)
#         return buffer, buffer_ptr, y_t 
    
#     def step(self, x_t, credit_matrix, buffer, buffer_ptr):
#         w_afferent, b_afferent = self.afferent.weight, self.afferent.bias
#         w_efferent, b_efferent = self.efferent.weight, self.efferent.bias
#         return ThinkingLearnableDelayRNN.step_jit(x_t, credit_matrix, buffer, buffer_ptr, self.lateral, self.max_delay, w_afferent, b_afferent, w_efferent, b_efferent)
    
#     def forward(self,
#                 x:Float[torch.Tensor, "batch_size time_steps input_size"],
#                 lengths:Int[torch.Tensor, "batch_size"],
#                 N=None) -> Seq2SeqOutput:
#         """
#         x shape: (batch_size, time_steps, input_size)
#         lengths shape: (batch_size,)
#         """
#         batch_size = x.size(0)
#         if N is None:
#             N = x.size(1) - 3  # N 유추

#         credit_matrix = self.calc_credit_matrix()
        
#         # 1. 초기화 및 인코딩(Encoding) 단계
#         # 전체 배치에 대한 buffer 초기화
#         buffer = x.new_zeros(self.max_delay + 1, batch_size, self.hidden_size)
#         buffer_ptr = 0 
        
#         # [핵심 로직 1] 패딩에 오염되기 전의 상태를 저장할 빈 텐서 준비
#         saved_buffers = buffer.new_zeros(buffer.size())
#         saved_buffer_ptrs = buffer.new_zeros(batch_size, dtype=torch.long)
        
#         # 입력 시퀀스를 순차적으로 읽으며 buffer에 문맥 축적
#         for t in range(x.size(1)):
#             x_t = x[:, t, :]
#             buffer, buffer_ptr, _ = self.step(x_t, credit_matrix, buffer, buffer_ptr)
            
#             # [핵심 로직 2] 방금 계산한 t가 어떤 배치의 '진짜 마지막 단어'인지 확인
#             is_last_token:torch.Tensor = (t == lengths - 1) # type: ignore # shape: (batch_size,)
            
#             if is_last_token.any():
#                 # 문장이 끝난 배치들만 현재의 깨끗한 buffer와 포인터를 복사해 둡니다.
#                 mask = is_last_token.view(1, batch_size, 1) # broadcasting을 위해 형태 변경
#                 saved_buffers = torch.where(mask, buffer, saved_buffers)
#                 saved_buffer_ptrs[is_last_token] = buffer_ptr
            
#         # 디코더의 첫 입력 (진짜 마지막 토큰 추출)
#         batch_indices = torch.arange(batch_size, out=x.new_empty(batch_size, dtype=torch.long))
#         dec_input = x[batch_indices, lengths - 1, :]
        
#         # 최종 결과를 저장할 텐서 (Logits 유지)
#         final_outputs = x.new_zeros(batch_size, N, self.vocab_size)
#         think_steps_list = []
        
#         # 배치별로 생각하는(Thinking) 시간이 다르므로 독립적으로 분리하여 연산
#         for i in range(batch_size):
#             # [핵심 로직 3] 루프를 끝까지 돌아 패딩에 오염된 원래 buffer 대신, 
#             # 아까 저장해둔 깨끗한 saved_buffers에서 상태를 불러와 생각을 시작합니다.
#             buffer_i = saved_buffers[:, i:i+1, :]     
#             buffer_ptr_i = saved_buffer_ptrs[i].item() # 저장된 정수형 포인터
#             curr_input = dec_input[i:i+1, :]
            
#             # --- 2. 생각(Thinking) 단계 ---
#             think_steps = 0
#             while True:
#                 # step 함수 호출
#                 buffer_i, buffer_ptr_i, y_t = self.step(curr_input, credit_matrix, buffer_i, buffer_ptr_i)
                
#                 # Gumbel-Softmax 샘플링 (미분 가능)
#                 curr_input = F.gumbel_softmax(y_t, tau=1.0, hard=True)
                
#                 sampled_idx = curr_input.argmax(dim=-1).item()
#                 think_steps += 1
                
#                 if sampled_idx == self.think_end_token or think_steps > self.max_think_steps:
#                     break  
#             think_steps_list.append(think_steps)
            
#             # --- 3. 출력(Output) 단계 ---
#             for j in range(N):
#                 buffer_i, buffer_ptr_i, y_t = self.step(curr_input, credit_matrix, buffer_i, buffer_ptr_i)
                
#                 # 정답 계산(Loss)을 위해 Logits 자체를 저장
#                 final_outputs[i, j, :] = y_t.squeeze(0)
                
#                 # 다음 스텝 입력을 위한 샘플링
#                 curr_input = F.gumbel_softmax(y_t, tau=1.0, hard=True)
                
#         return Seq2SeqOutput(outputs=final_outputs[:,:,:-1], think_steps=torch.tensor(think_steps_list, device=self.device))

class FastThinkingLearnableDelayRNN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_classes:int, max_delay:int, config:Config):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_think_steps = config.max_think_steps
        self.config = config
        
        # 딜레이가 0이면 일반 FFN과 다름없으므로 최소 1 이상이어야 함
        assert max_delay >= 1, "max_delay must be at least 1"
        self.max_delay = max_delay
        self.device = config.device
        
        self.vocab_size = num_classes + 1
        self.think_end_token = num_classes
        self.output_size = self.vocab_size
        
        self.afferent = nn.Linear(self.input_size, hidden_size)
        self.lateral = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(hidden_size, hidden_size))
        )
        self.efferent = nn.Linear(hidden_size, self.output_size)
        
        self.tau = nn.Parameter(max_delay * torch.rand_like(self.lateral) + 1)
        self.scale_exponent = nn.Parameter(torch.zeros(hidden_size, device=self.device))

    @staticmethod
    @torch.jit.script
    def calc_credit_matrix_jit(tau_clipped:torch.Tensor, max_delay:int, hidden_size:int, scale_exponent:torch.Tensor) -> torch.Tensor:
        # 기존과 동일
        credit_matrix = torch.arange(max_delay + 1, out=tau_clipped.new_empty(max_delay + 1)) 
        credit_matrix = credit_matrix[:, None, None]
        distance = 1.0 + torch.abs(credit_matrix - tau_clipped)

        raw_credit = distance.pow(-softplus(scale_exponent[None, :, None]))
        differentiable = raw_credit
        # differentiable = raw_credit / (raw_credit.sum(dim=0, keepdim=True)) # Backward pass goes through this path
        
        nondifferentiable = (credit_matrix == tau_clipped).float() # Forward pass uses this hard assignment for stability and interpretability
        
        # Straight-Through Estimator trick
        # Return equals a spike in the tau_clipped position,
        # but gradients flow through the differentiable soft assignment, enabling learning of tau.
        return differentiable + (nondifferentiable - differentiable).detach()
    
    def calc_credit_matrix(self): 
        return FastThinkingLearnableDelayRNN.calc_credit_matrix_jit(
            torch.clamp(self.tau, 1, self.max_delay)[None,...],
            self.max_delay,
            self.hidden_size,
            self.scale_exponent
        )
    
    def _adjust_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gumbel-Softmax 출력(vocab_size)을 입력(input_size) 규격으로 자동 패딩/슬라이싱"""
        if tensor.size(-1) != self.input_size:
            diff = self.input_size - tensor.size(-1)
            if diff > 0:
                tensor = F.pad(tensor, (0, diff))
            else:
                tensor = tensor[..., :self.input_size]
        return tensor

    def step_fast(self, x_t, history, ptr, W_rev):
        """
        [핵심 1] O(1) Memory Write의 Gather(Pull) 스텝 함수
        x_t: (batch, input_size)
        history: (max_delay, batch, hidden_size) - 과거의 h_to_delay 기록
        W_rev: (max_delay, hidden_size, hidden_size) - 뒤집힌 가중치 행렬
        """
        # 1. 포인터 위치에 맞춰 가중치 W를 가볍게 회전 (B 차원이 없어서 압도적으로 빠름)
        W_aligned = torch.roll(W_rev, shifts=ptr, dims=0)
        
        # 2. 과거 히스토리와 가중치를 곱해 현재 시점에 도달한 지연 신호 합산 (Gather)
        # d: delay 시간, h: hidden_out, i: hidden_in, b: batch
        h_delayed = torch.einsum('dhi,dbi->bh', W_aligned, history)
        
        # 3. 새로운 은닉 상태 계산
        h_to_delay = torch.tanh(self.afferent(x_t) + h_delayed)
        
        # 4. 버퍼 업데이트 (Autograd in-place 에러를 피하기 위해 torch.where 사용)
        mask = (torch.arange(self.max_delay, device=x_t.device) == ptr).view(-1, 1, 1)
        history = torch.where(mask, h_to_delay.unsqueeze(0), history)
        
        # 5. 포인터 이동 및 출력 계산
        ptr = (ptr + 1) % self.max_delay
        y_t = self.efferent(h_delayed)
        
        return history, ptr, y_t

    def forward(self, x, lengths, N=None, targets=None, teacher_forcing_ratio=0.0) -> Seq2SeqOutput:
        batch_size = x.size(0)
        device = x.device
        if N is None:
            N = x.size(1) - 3  

        # --- 사전 연산 (Pre-computation) ---
        credit_matrix = self.calc_credit_matrix()
        # 원래 코드에서 credit_matrix[0]은 buffer_ptr에서 읽히자마자 (~mask)로 지워졌음. 
        # 즉 미래에 아무 영향도 주지 않는 값이므로 [1:]부터 잘라 쓰는 것이 수학적으로 완벽히 동일함.
        W = credit_matrix[1:] * self.lateral[None, :, :]  # (max_delay, hidden, hidden)
        W_rev = W.flip(0)  # 과거 기록과 매칭하기 위해 미리 뒤집어 둠
        
        # --- 1. 인코딩(Encoding) 단계 ---
        history = x.new_zeros(self.max_delay, batch_size, self.hidden_size)
        ptr = 0 
        
        saved_history = history.new_zeros(history.size()) # (max_delay, batch_size, hidden_size)
        saved_ptrs = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            history, ptr, _ = self.step_fast(x_t, history, ptr, W_rev)
            
            is_last_token = (t == lengths - 1) 
            if is_last_token.any():
                mask = is_last_token.view(1, batch_size, 1) 
                saved_history = torch.where(mask, history, saved_history)
                saved_ptrs = torch.where(is_last_token, torch.tensor(ptr, device=device), saved_ptrs)
            
        # --- Alignment 트릭 (Thinking 시작 전 포인터 동기화) ---
        idx = torch.arange(self.max_delay, device=device).unsqueeze(1) # (max_delay, 1)
        gather_idx = (idx + saved_ptrs.unsqueeze(0)) % self.max_delay # (max_delay, batch_size)
        gather_idx = gather_idx.unsqueeze(-1).expand(self.max_delay, batch_size, self.hidden_size) # (max_delay, batch_size, hidden_size)
        
        # 모든 배치의 히스토리를 정렬하여 포인터를 0으로 통일
        aligned_history = torch.gather(saved_history, 0, gather_idx) # (max_delay, batch_size, hidden_size)
        
        # --- 2. 생각(Thinking) 단계 ---
        history = aligned_history
        ptr = 0  # 이제 배치 전체가 완벽하게 동기화됨
        
        batch_indices = torch.arange(batch_size, device=device)
        curr_input = x[batch_indices, lengths - 1, :]
        
        is_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        think_steps_tensor = torch.zeros(batch_size, dtype=torch.long, device=device)
        think_steps = 0
        
        last_valid_input = curr_input.clone()
        
        while not is_done.all() and think_steps < self.max_think_steps:
            history, ptr, y_t = self.step_fast(curr_input, history, ptr, W_rev)
            
            curr_input_sampled = F.gumbel_softmax(y_t, tau=1.0, hard=True)
            curr_input_adj = self._adjust_dim(curr_input_sampled)
            
            sampled_idx = curr_input_sampled.argmax(dim=-1)
            just_finished = (sampled_idx == self.think_end_token) & ~is_done
            
            # 출력 단계(Output Phase)의 첫 입력으로 쓰기 위해 마지막으로 정상적인 토큰 캐싱
            last_valid_input = torch.where((~is_done).unsqueeze(-1), curr_input_adj, last_valid_input)
            
            think_steps_tensor += (~is_done).long()
            is_done = is_done | just_finished
            think_steps += 1
            
            # [핵심 2] Idling: 끝난 배치는 입력을 0으로 주입 (SNN의 휴지기 시뮬레이션)
            curr_input = curr_input_adj * (~is_done).unsqueeze(-1).to(curr_input_adj.dtype)
            
        # --- 3. 출력(Output) 단계 ---
        curr_input = last_valid_input
        final_outputs = x.new_zeros(batch_size, N, self.vocab_size)
        
        # for j in range(N):
        #     history, ptr, y_t = self.step_fast(curr_input, history, ptr, W_rev)
        #     final_outputs[:, j, :] = y_t
        #     one_hot_out = F.gumbel_softmax(y_t, tau=1.0, hard=True)
        #     curr_input = self._adjust_dim(curr_input_sampled)
            
        #     true_token_idx = targets[i, j]
        #     is_teacher_forcing = (
        #         (targets is not None) and 
        #         (torch.rand(1).item() < teacher_forcing_ratio) and 
        #         (true_token_idx >= 0)  # <-- 핵심: -1(패딩)이 아닐 때만!
        #     )
            
        #     if is_teacher_forcing:
        #         # 정답 인덱스를 가져와서 One-hot 인코딩 후 shape 변환 (1, 1, vocab_size)
        #         true_token_idx = targets[i, j]
        #         curr_input = F.one_hot(true_token_idx, num_classes=self.vocab_size).float().view(1, 1, -1)
        #     else:
        #         # 기존 방식: 모델이 예측한 값을 다음 스텝의 입력으로 사용
        #         curr_input = one_hot_out.unsqueeze(1)
        
        for j in range(N): 
            # curr_input: (batch, input_size) -> step_fast -> y_t: (batch, vocab_size)
            history, ptr, y_t = self.step_fast(curr_input, history, ptr, W_rev)
            final_outputs[:, j, :] = y_t
            
            # 모델의 예측값 (Batch, Vocab_size)
            one_hot_out = F.gumbel_softmax(y_t, tau=1.0, hard=True)
            
            # -----------------------------------------------------------------
            # [수정된 배치 단위 교사 강제(Teacher Forcing) 로직]
            # -----------------------------------------------------------------
            if targets is not None:
                # 1. j번째 스텝의 정답 (Batch,)
                targets_j = targets[:, j]
                
                # 2. 패딩(-1)이 아닌 유효한 타겟인지 확인하는 마스크 (Batch,)
                valid_mask = (targets_j >= 0)
                
                # 3. 교사 강제 확률 주사위 (Batch,) - 각 샘플마다 개별적으로 주사위 굴림
                tf_dice = (torch.rand(targets_j.size(0), device=targets.device) < teacher_forcing_ratio)
                
                # 4. 최종적으로 교사 강제를 적용할 마스크 (Batch,)
                do_tf_mask = valid_mask & tf_dice
                
                # 5. 정답 One-hot 인코딩 (-1 에러 방지를 위해 clamp로 -1을 0으로 덮어씌움)
                # (어차피 do_tf_mask가 False인 곳은 이 값을 쓰지 않으므로 안전합니다)
                safe_targets_j = torch.clamp(targets_j, min=0) # (Batch,)
                true_one_hot = F.one_hot(safe_targets_j, num_classes=self.vocab_size).float() # (Batch, Vocab_size)
                
                # 6. 차원 맞추기 (Batch, Vocab_size) -> (Batch, 1, Vocab_size) 같은 형태를 위해
                # do_tf_mask를 (Batch, 1)로 늘려줌
                do_tf_mask = do_tf_mask.unsqueeze(-1)
                
                # 7. torch.where를 사용해 조건에 따라 정답(true_one_hot) 또는 예측값(one_hot_out) 선택
                curr_input = torch.where(do_tf_mask, true_one_hot, one_hot_out) # (Batch, Vocab_size)
            else:
                # 추론(Test) 시에는 예측값만 사용
                curr_input = one_hot_out
                
        return Seq2SeqOutput(outputs=final_outputs[:,:,:-1], think_steps=think_steps_tensor)

if __name__ == "__main__":
    # Example usage
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = get_model(ModelType.RNN, device=torch.device("cpu"), config=presets[ModelType.RNN])
    print("ThinkingRNN", count_parameters(model), "parameters")

    model = get_model(ModelType.LSTM, device=torch.device("cpu"), config=presets[ModelType.LSTM])
    print("ThinkingLSTM", count_parameters(model), "parameters")

    model = get_model(ModelType.GRU, device=torch.device("cpu"), config=presets[ModelType.GRU])
    print("ThinkingGRU", count_parameters(model), "parameters")
    
    model = get_model(ModelType.DelayedRNN, device=torch.device("cpu"), config=presets[ModelType.DelayedRNN])
    print("ThinkingLearnableDelayRNN", count_parameters(model), "parameters")