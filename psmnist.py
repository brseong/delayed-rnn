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
from utils.model.classification import get_model_with_preset, SimpleRNN, SimpleLSTM, SimpleGRU, SimpleTransformer, LearnableDelayRNN
from utils.io import save_model, load_model
from tqdm.auto import tqdm
from time import time

# torch.autograd.set_detect_anomaly(True)

config = Config.from_args(get_args())
model = get_model_with_preset(config.model_type)
model = model.to(config.device)

run = wandb.init(entity="CIDA", project="PSMNIST_RNN", name=f"{config.model_type.name}", config=asdict(config))
run.__enter__()

print(f"Using device: {config.device}")

# %%
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

# 3. 고정된 순열(Permutation) 생성
# 모든 배치와 에폭에서 동일한 순서로 섞어야 PSMNIST가 성립됩니다.
if config.seed is not None:
    torch.manual_seed(config.seed) # 재현성을 위해 시드 고정
perm_order = torch.randperm(config.seq_length).to(config.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * len(train_loader))

run.define_metric("Accuracy/Validation", step_metric="val_step")
run.define_metric("Time/Validation", step_metric="val_step")
val_step = 0

best_val_acc = 0.0
# 5. 학습 루프
for epoch in tqdm(range(config.epochs), desc="Epochs"):
    model.train()
    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Batches", leave=False):
        # 이미지 변형: (Batch, 1, 28, 28) -> (Batch, 784, 1)
        images = images.view(-1, config.seq_length, config.input_size).to(config.device)
        labels = labels.to(config.device)
        
        # *** 중요: 여기서 픽셀 순서를 섞습니다 (PSMNIST 핵심) ***
        images = images[:, perm_order, :]
        
        optimizer.zero_grad()
        
        t_start = time()
        
        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 역전파 및 최적화
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        dt = time() - t_start
        
        accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()
        
        wandb.log({"Loss/Train": loss.item(),
                   "Accuracy/Train": accuracy,
                   "Time/Train": dt,
                   "Learning_Rate": scheduler.get_last_lr()[0]})
        if isinstance(model, LearnableDelayRNN):
            wandb.log({"Scale_Exponent/Bias": softplus(model.scale_exponent.data).mean().item(),
                       "Scale_Exponent/Variance": softplus(model.scale_exponent.data).var().item()})
        if (i+1) % 300 == 0:
            print(f'Epoch [{epoch+1}/{config.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {accuracy:.4f}')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        dt_total = 0
        for images, labels in tqdm(test_loader, desc="Validation Batches", leave=False):
            images = images.view(-1, config.seq_length, config.input_size).to(config.device)
            
            # 테스트셋에도 동일한 순열 적용
            images = images[:, perm_order, :]
            labels = labels.to(config.device)
            
            t_start = time()
            outputs = model(images)
            dt_total += time() - t_start
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        if acc > best_val_acc:
            best_val_acc = acc
            save_model(model, asdict(config) | {"best_val_accuracy": acc, "epoch": epoch})
        
        wandb.log({"val_step": (val_step := val_step + 1),
                   "Accuracy/Validation": acc,
                   "Time/Validation": dt_total / len(test_loader)})
        print(f'Validation Accuracy after Epoch {epoch+1}: {acc:.2f}%')
# 6. 평가 루프
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    dt_total = 0
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.view(-1, config.seq_length, config.input_size).to(config.device)
        
        # 테스트셋에도 동일한 순열 적용
        images = images[:, perm_order, :]
        labels = labels.to(config.device)
        
        t_start = time()
        outputs = model(images)
        dt_total += time() - t_start
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    wandb.log({"Accuracy/Test": 100 * correct / total,
               "Time/Test": dt_total / len(test_loader)})
    print(f'Test Accuracy of the RNN on the 10000 test images (PSMNIST): {100 * correct / total:.2f}%')

# %%
run.__exit__(None, None, None)


