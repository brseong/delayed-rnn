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
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * len(train_loader))
scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0) # 학습 안정화 시 CosineAnnealingLR로 대체

best_val_acc = 0.0
best_model_state = None
# 5. 학습 루프
for epoch in tqdm(range(config.epochs), desc="Epochs"):
    total_loss = 0
    model.train()
    for i, (inputs, targets, lengths) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Batches", leave=False):
        inputs, targets, lengths = inputs.to(config.device), targets.to(config.device), lengths.to(config.device)
        
        optimizer.zero_grad()
        
        # 모델 Forward
        seq2seq_out:Seq2SeqOutput = model(inputs, lengths, N=targets.size(1))
        outputs = seq2seq_out.outputs  # [Batch, Max_Len, K]
        think_steps = seq2seq_out.think_steps  # [Batch]
        
        # Loss
        # outputs: [Batch, Max_Len, K] -> Flatten
        # targets: [Batch, Max_Len] -> Flatten (-1은 ignore_index 처리됨)
        # loss = criterion(outputs.reshape(config.batch_size, config.seq_max, config.num_classes), targets.reshape(-1))
        valid_mask = targets != -1
        outputs_flat = outputs[valid_mask].view(-1, config.num_classes) 
        targets_flat = targets[valid_mask]
        
        
        loss = criterion(outputs_flat, targets_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            valid_mask = targets_flat != -1
            if valid_mask.sum() > 0:
                predictions = outputs_flat.argmax(dim=-1)
                correct_predictions = (predictions[valid_mask] == targets_flat[valid_mask]).float()
                accuracy = correct_predictions.mean().item()
            else:
                accuracy = 0.0
        
        wandb.log({"Loss/Train": loss.item(),
                "Accuracy/Train": accuracy,
                "Think_Steps/Train": think_steps.float().mean().item(),
                "Learning_Rate": scheduler.get_last_lr()[0]})
        if isinstance(model, FastThinkingLearnableDelayRNN):
            wandb.log({"Scale_Exponent/Bias": softplus(model.scale_exponent.data).mean().item(),
                        "Scale_Exponent/Variance": softplus(model.scale_exponent.data).var().item()})
                
        if (i+1) % 300 == 0:
            print(f'Epoch [{epoch+1}/{config.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {accuracy:.4f}')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        think_steps_list = []
        for inputs, targets, lengths in tqdm(val_loader, desc="Validation Batches", leave=False):
            inputs, targets, lengths = inputs.to(config.device), targets.to(config.device), lengths.to(config.device)
            
            seq2seq_out:Seq2SeqOutput = model(inputs, lengths, N=targets.size(1))
            outputs = seq2seq_out.outputs  # [Batch, Max_Len, K]
            think_steps = seq2seq_out.think_steps  # [Batch]
            
            _, predicted = torch.max(outputs.data, 2)
            total += targets.size(0)
            correct += torch.all(predicted == targets, dim=1).sum().item()
            think_steps_list.extend(think_steps.tolist())
        
        acc = 100 * correct / total
        if acc > best_val_acc:
            best_val_acc = acc
            save_model(model, asdict(model.config) | {"best_val_accuracy": acc, "epoch": epoch})
        
        wandb.log({"Accuracy/Validation": 100 * correct / total,
                   "Think_Steps/Validation": sum(think_steps_list) / len(think_steps_list) if think_steps_list else 0})
        print(f'Validation Accuracy after Epoch {epoch+1}: {100 * correct / total:.2f}%')
            
# 6. 평가 루프
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    think_steps_list = []
    for inputs, targets, lengths in tqdm(test_loader, desc="Testing"):
        inputs, targets, lengths = inputs.to(config.device), targets.to(config.device), lengths.to(config.device)
        
        seq2seq_out:Seq2SeqOutput = model(inputs, lengths, N=targets.size(1))
        outputs = seq2seq_out.outputs  # [Batch, Max_Len, K]
        think_steps = seq2seq_out.think_steps  # [Batch]
        
        _, predicted = torch.max(outputs.data, 2)
        total += targets.size(0)
        correct += torch.all(predicted == targets, dim=1).sum().item()
        think_steps_list.extend(think_steps.tolist())
        
    wandb.log({"Accuracy/Test": 100 * correct / total,
                "Think_Steps/Test": sum(think_steps_list) / len(think_steps_list) if think_steps_list else 0})

    print(f'Test Accuracy of the RNN on the 10000 test images (QSWAP): {100 * correct / total:.2f}%')

# %%
run.__exit__(None, None, None)


