# %%
from dataclasses import dataclass, asdict
from pathlib import Path
import torch, wandb
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softplus
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.config import Config, ModelType, get_args
from utils.model.seq2seq import get_model, Seq2SeqOutput, FastThinkingLearnableDelayRNN
from utils.data import SwapDataset, collate_fn
from utils.io import save_model, load_model
from tqdm.auto import tqdm
from time import time

# torch.autograd.set_detect_anomaly(True)

config = Config.from_args(get_args())
model = get_model(config.model_type, config.device, config=config)

run = wandb.init(entity="CIDA", project="QSWAP_RNN", name=f"{config.model_type.name}", config=asdict(config))
run.__enter__()

print(f"Using device: {config.device}")

# %%
DATASET_SIZE = 10000 # 한 Epoch에 사용할 데이터 수
train_dataset = SwapDataset(size=DATASET_SIZE, k=config.input_size-1, min_len=config.seq_min, max_len=config.seq_max)
val_dataset = SwapDataset(size=DATASET_SIZE//10, k=config.input_size-1, min_len=config.seq_min, max_len=config.seq_max)
test_dataset = SwapDataset(size=DATASET_SIZE//10, k=config.input_size-1, min_len=config.seq_min, max_len=config.seq_max)

# DataLoader 생성 (collate_fn 등록 필수!)
train_loader = DataLoader(
    train_dataset, 
    batch_size=config.batch_size,       # 확인용으로 작은 배치
    shuffle=True,       # 학습 시 셔플 추천
    collate_fn=collate_fn # 우리가 만든 패딩 함수 적용
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=config.batch_size, 
    shuffle=False, 
    collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=config.batch_size, 
    shuffle=False, 
    collate_fn=collate_fn
)

# %%
if config.seed is not None:
    torch.manual_seed(config.seed) # 재현성을 위해 시드 고정
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * len(train_loader))
# scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0) # 학습 안정화 시 CosineAnnealingLR로 대체


def tokenwise_accuracy_from_flat_logits(outputs_flat: torch.Tensor, targets_flat: torch.Tensor) -> float: #(예측 점수, 정답)
    if targets_flat.numel() == 0:
        return 0.0
    predictions = outputs_flat.argmax(dim=-1) # 예측값 추려내기
    return (predictions == targets_flat).float().mean().item() # 예측과 정답이 같은지 확인 -> 1,0으로 반환 -> 평균 구하기 -> float으로 변환

run.define_metric("Accuracy/Validation", step_metric="val_step") # (y축, x축)
run.define_metric("Accuracy/Validation_Tokenwise", step_metric="val_step")
run.define_metric("Think_Steps/Validation", step_metric="val_step")
run.define_metric("Time/Validation", step_metric="val_step")
run.define_metric("Accuracy/Test", step_metric="test_step")
run.define_metric("Accuracy/Test_Tokenwise", step_metric="test_step")
run.define_metric("Think_Steps/Test", step_metric="test_step")
run.define_metric("Time/Test", step_metric="test_step")
val_step, test_step = 0, 0

best_val_acc = 0.0
best_model_state = None
# 5. 학습 루프
for epoch in tqdm(range(config.epochs), desc="Epochs"):
    total_loss = 0
    tf_ratio = max(0.0, 1.0 - (epoch / config.epochs)) if config.teach_forcing else 0.0 # 처음에는 정답만 알려주고 그 비율 줄여감
    scale_exponent = 0.5 * (1 + 5 * epoch / config.epochs)
    model.train()
    for i, (inputs, targets, lengths) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Batches", leave=False):
        inputs, targets, lengths = inputs.to(config.device), targets.to(config.device), lengths.to(config.device)
        
        optimizer.zero_grad()
        
        t_start = time()
        
        # 모델 Forward
        seq2seq_out:Seq2SeqOutput = model(inputs, lengths, N=targets.size(1), targets=targets, teacher_forcing_ratio=tf_ratio)
        outputs = seq2seq_out.outputs  # [Batch, Max_Len, K]
        think_steps = seq2seq_out.think_steps  # [Batch]
        
        # Loss
        # outputs: [Batch, Max_Len, K] -> Flatten
        # targets: [Batch, Max_Len] -> Flatten (-1은 ignore_index 처리됨)
        # loss = criterion(outputs.reshape(config.batch_size, config.seq_max, config.num_classes), targets.reshape(-1))
        valid_mask = targets != -1 # 패딩 없애기 : 진짜 데이터만 골라내기
        outputs_flat = outputs[valid_mask].view(-1, config.num_classes) 
        targets_flat = targets[valid_mask]
        
        loss = criterion(outputs_flat, targets_flat) # crossentropy로 오차 계산
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 기울기 폭발 방지
        optimizer.step() # 가중치 업데이트
        scheduler.step() # 학습률 업데이트
        
        dt = time() - t_start
        
        total_loss += loss.item()
        
        with torch.no_grad():
            accuracy = tokenwise_accuracy_from_flat_logits(outputs_flat, targets_flat)
        
        wandb.log({"Loss/Train": loss.item(),
                "Accuracy/Train": accuracy,
                "Think_Steps/Train": think_steps.float().mean().item(),
                "Time/Train": dt,
                "Time_Per_Step/Train": dt / (2 * inputs.size(1) + think_steps.max().item()),  # 대략적인 시간/스텝 계산
                "Learning_Rate": scheduler.get_last_lr()[0]})
        if isinstance(model, FastThinkingLearnableDelayRNN):
            if isinstance(model.scale_exponent, torch.nn.Parameter):
                wandb.log({"Scale_Exponent/Bias": softplus(model.scale_exponent.data).mean().item(),
                            "Scale_Exponent/Variance": softplus(model.scale_exponent.data).var().item()})
            elif isinstance(model.scale_exponent, float):
                wandb.log({"Scale_Exponent/Value": model.scale_exponent})
                model.scale_exponent = scale_exponent
            else:
                raise ValueError("Unsupported type for scale_exponent in FastThinkingLearnableDelayRNN")
            
                
        if (i+1) % 300 == 0: # 300번 배치마다 현재 상태 출력 -> 학습 잘 되고 있는지 확인
            print(f'Epoch [{epoch+1}/{config.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {accuracy:.4f}')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        dt_total = 0
        for inputs, targets, lengths in tqdm(val_loader, desc="Validation Batches", leave=False):
            inputs, targets, lengths = inputs.to(config.device), targets.to(config.device), lengths.to(config.device)
            
            t_start = time()
            
            seq2seq_out:Seq2SeqOutput = model(inputs, lengths, N=targets.size(1))
            outputs = seq2seq_out.outputs  # [Batch, Max_Len, K]
            think_steps = seq2seq_out.think_steps  # [Batch]
            
            dt_total += time() - t_start
            
            _, predicted = torch.max(outputs.data, 2)
            total += targets.size(0)
            correct += torch.all(predicted == targets, dim=1).sum().item()
            
            valid_mask = targets != -1
            outputs_flat = outputs[valid_mask].view(-1, config.num_classes) # flatten
            targets_flat = targets[valid_mask]
            with torch.no_grad():
                accuracy = tokenwise_accuracy_from_flat_logits(outputs_flat, targets_flat)
                wandb.log({"val_step": (val_step := val_step + 1),
                           "Accuracy/Validation_Tokenwise": accuracy,
                           "Think_Steps/Validation": think_steps.float().mean().item()})
        
        acc = 100 * correct / total
        if acc > best_val_acc:
            best_val_acc = acc
            save_model(model, asdict(model.config) | {"best_val_accuracy": acc, "epoch": epoch})
        
        wandb.log({"Accuracy/Validation": 100 * correct / total,
                   "Time/Validation": dt_total/len(val_loader)})
        print(f'Validation Accuracy after Epoch {epoch+1}: {100 * correct / total:.2f}%')
            
# 6. 평가 루프
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    dt_total = 0
    think_steps_list = []
    for inputs, targets, lengths in tqdm(test_loader, desc="Testing"):
        inputs, targets, lengths = inputs.to(config.device), targets.to(config.device), lengths.to(config.device)
        
        t_start = time()
        seq2seq_out:Seq2SeqOutput = model(inputs, lengths, N=targets.size(1))
        outputs = seq2seq_out.outputs  # [Batch, Max_Len, K]
        think_steps = seq2seq_out.think_steps  # [Batch]
        dt_total += time() - t_start
        
        _, predicted = torch.max(outputs.data, 2)
        total += targets.size(0)
        correct += torch.all(predicted == targets, dim=1).sum().item()
        think_steps_list.extend(think_steps.tolist())
        
        valid_mask = targets != -1
        outputs_flat = outputs[valid_mask].view(-1, config.num_classes) 
        targets_flat = targets[valid_mask]
        with torch.no_grad():
            accuracy = tokenwise_accuracy_from_flat_logits(outputs_flat, targets_flat)
            wandb.log({"test_step": (test_step := test_step + 1),
                       "Accuracy/Test_Tokenwise": accuracy,
                       "Think_Steps/Test": think_steps.float().mean().item()})
        
    wandb.log({"Accuracy/Test": 100 * correct / total,
                "Time/Test": dt_total/len(test_loader)})

    print(f'Test Accuracy of the RNN on the 10000 test images (QSWAP): {100 * correct / total:.2f}%')

# %%
run.__exit__(None, None, None)


