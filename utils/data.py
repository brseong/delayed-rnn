import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random

class SwapDataset(Dataset):
    def __init__(self, size, k = 10, min_len = 5, max_len = 20):
        self.size = size
        self.k = k
        self.min_len = min_len
        self.max_len = max_len

    def __len__(self):
        return self.size

    def _int_to_binary_vec(self, n):
        """정수를 K차원 이진수 벡터로 변환"""
        # 예: 3 -> '00...011' -> [0, 0, ..., 1, 1]
        binary_str = format(n, f'0{self.k}b')
        # 길이가 K를 넘어가면 잘라냄 (Error handling)
        if len(binary_str) > self.k: 
            binary_str = binary_str[-self.k:]
        return [float(b) for b in binary_str]

    def __getitem__(self, idx):
        # 1. 랜덤 길이 설정
        seq_len = random.randint(self.min_len, self.max_len)
        
        # 2. 데이터 시퀀스 생성 (One-hot Indices)
        seq_indices = [random.randint(0, self.k - 1) for _ in range(seq_len)]
        
        # 3. Swap 위치 선정
        pos1, pos2 = random.sample(range(seq_len), 2)
        
        # --- 입력 구성 (Vectors) ---
        input_vectors = []
        
        # (1) Data Part: One-hot vectors
        for idx in seq_indices:
            vec = [0.0] * (self.k + 1)
            vec[idx] = 1.0 
            input_vectors.append(vec)
            
        # (2) SEP Token: Last dim = 1
        sep_vec = [0.0] * (self.k + 1)
        sep_vec[self.k] = 1.0
        input_vectors.append(sep_vec)
        
        # (3) Command Part: Binary Position Vectors
        # Pos1
        cmd1_vec = self._int_to_binary_vec(pos1) + [0.0] # SEP dim은 0
        input_vectors.append(cmd1_vec)
        # Pos2
        cmd2_vec = self._int_to_binary_vec(pos2) + [0.0]
        input_vectors.append(cmd2_vec)
        
        # --- 정답 구성 (Indices) ---
        target_seq = seq_indices[:]
        target_seq[pos1], target_seq[pos2] = target_seq[pos2], target_seq[pos1]
        
        # 텐서 변환
        # Input: [Seq_Len + 3, Input_Dim]
        # Target: [Seq_Len]
        return (torch.tensor(input_vectors, dtype=torch.float32), 
                torch.tensor(target_seq, dtype=torch.long),
                seq_len)

def collate_fn(batch):
    """
    배치 내의 샘플들을 받아서 Padding을 수행하고 텐서로 합침
    batch: List of (input_tensor, target_tensor, length)
    """
    inputs, targets, lengths = zip(*batch)
    
    # 1. Input Padding (뒤에 0으로 채움)
    # batch_first=True: [Batch, Max_Len, Dim] 형태로 만듦
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    
    # 2. Target Padding (뒤에 -1로 채움 -> Loss 계산 시 무시)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=-1)
    
    # 3. Lengths는 Tensor로 변환
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return padded_inputs, padded_targets, lengths