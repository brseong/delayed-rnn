import time
import wandb
import hydra
import torch 

torch.set_float32_matmul_precision('high')

from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from utils import set_seed

from utils import (
    TRAIN_LOSS,
    TRAIN_TOKEN_ACC,
    TRAIN_SEQ_ACC,
    EVAL_LOSS,
    EVAL_TOKEN_ACC,
    EVAL_SEQ_ACC,
    EVAL_FLOPS_1B,
)

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(args):
    print("-"*20, "Experiment Configuration", "-"*20)
    print(OmegaConf.to_yaml(args))
    print("-"*60)
    
    seed: int = args.seed
    batch_size: int = args.batch_size
    num_workers: int = args.num_workers
    hidden_size: int = args.model_args.hidden_size
    num_layers: int = args.model_args.num_layers  
    num_epochs: int = args.num_epochs
    
    lr: float = args.lr
    
    model_dict: str = args.model
    dataset_name: str = args.dataset
    group_name: str = args.wandb.group_name
    
    use_wandb: bool = args.use_wandb
    use_lr_scheduler: bool = args.use_lr_scheduler
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed = seed)
    
    train_dataset = instantiate(dataset_name, train = True, seed = seed)
    eval_dataset = instantiate(dataset_name, train = False, seed = seed)
    
    input_size: int = train_dataset.input_size # input vector size (feature dimension)
    num_classes: int = train_dataset.num_classes # if dataset has attribute num_classes, use it. Otherwise, default to 0.
    dataset_name: str = train_dataset.dataset_name
    is_classification: bool = train_dataset.is_classification
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        collate_fn = train_dataset.get_collate_fn() if hasattr(train_dataset, 'get_collate_fn') else None
    )

    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        collate_fn = eval_dataset.get_collate_fn() if hasattr(eval_dataset, 'get_collate_fn') else None
    )
    
    model = instantiate(
        model_dict, 
        input_size=input_size, 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        num_classes=num_classes,
        batch_size=batch_size,
        is_classification=is_classification,
        device=device,
    )
    
    # model = torch.compile(model)

    model_name: str = model.model_name

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}, Dataset: {dataset_name}, Num Params: {num_params}")
    
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.1*lr)
        
    if use_wandb: 
        unique_id = f"{model_name}_{group_name}_{dataset_name}_{seed}_{time.time()}"
        wandb.init(
            entity=args.wandb.entity,
            project=dataset_name, 
            id=unique_id,
            name=f"{model_name}_{seed}",
            group=group_name,
            config=OmegaConf.to_container(args, resolve=True),
        )
        wandb.config.update({"num_params": num_params})

    for epoch in tqdm(range(num_epochs), desc=f"Epochs [{dataset_name}]"):
        
        train_logs = train_dataset.train(
            model = model, 
            dataloader = train_dataloader, 
            optimizer = optimizer,
            device = device,
        )
        
        if use_lr_scheduler:
            scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - \
            Loss: {train_logs[TRAIN_LOSS]:.4f}, \
            Token Accuracy: {train_logs[TRAIN_TOKEN_ACC]:.4f}, \
            Sequence Accuracy: {train_logs[TRAIN_SEQ_ACC]:.4f}"
        )
        
        eval_logs = eval_dataset.eval(
            model = model,
            dataloader = eval_dataloader,
            device = device
        )
        
        print(f"Eval - Loss: {eval_logs[EVAL_LOSS]:.4f}, \
            Token Accuracy: {eval_logs[EVAL_TOKEN_ACC]:.4f}, \
            Sequence Accuracy: {eval_logs[EVAL_SEQ_ACC]:.4f}, \
        FLOPs (1B): {eval_logs[EVAL_FLOPS_1B]:.2f}")
        
        if use_wandb:
            wandb.log({
                **train_logs,
                **eval_logs,
            })
            
        
        

if __name__ == "__main__":
    main()