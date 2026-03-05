import time 
import torch
import hydra
import wandb

from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from hydra.utils import instantiate

from utils import set_seed, calculate_flops, init_model_compile

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(args):
    print("-"*20, "Experiment Configuration", "-"*20)
    print(OmegaConf.to_yaml(args))
    print("-"*60)
    
    seed: int = args.seed
    num_epochs: int = args.num_epochs
    batch_size: int = args.batch_size
    hidden_size: int = args.hidden_size
    lr: float = args.lr
    
    model_name: str = args.model
    dataset_name: str = args.dataset
    group_name: str = args.wandb.group_name
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    use_wandb: bool = args.wandb.use_wandb
    use_lr_scheduler: bool = args.use_lr_scheduler
    use_model_compile: bool = args.use_model_compile
    
    set_seed(seed = seed)
    
    train_dataset = instantiate(dataset_name, train = True)
    eval_dataset = instantiate(dataset_name, train = False)
    
    input_size: int = train_dataset.input_size
    output_size: int = train_dataset.output_size 
    job_type: str = train_dataset.job_type 
    dataset_name: str = train_dataset.dataset_name
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True,
        collate_fn = train_dataset.get_collate_fn() if hasattr(train_dataset, 'get_collate_fn') else None
    )

    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True,
        collate_fn = eval_dataset.get_collate_fn() if hasattr(eval_dataset, 'get_collate_fn') else None
    )
    
    model = instantiate(
        model_name,
        input_size=input_size, 
        hidden_size=hidden_size, 
        output_size=output_size,
        batch_size=batch_size,
        job_type=job_type, 
        device=device,
    )
    with torch.no_grad():
        flops_million = calculate_flops(
            model=model, 
            batch_size=batch_size,
            input_size=input_size,
            device=device,
            job_type=job_type,
        )
    
    model = torch.compile(
        model = model,
        mode="reduce-overhead",
        fullgraph=True,
        disable=use_model_compile
    )
    
    if use_model_compile:
        start_time = time.time()
        init_model_compile(
            model,
            eval_dataloader,
            device,
            job_type
        )
        compile_time = time.time() - start_time
    
    model_name: str = model.model_name
    num_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
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
        
        wandb.config.update({"Flops(1M)": flops_million})
        wandb.config.update({"num_params": num_params})
        wandb.config.update({"compile_time": compile_time if use_model_compile else None})

    print(f"Model: {model_name}, Dataset: {dataset_name}, Num Params: {num_params}, Flops(1M): {flops_million}, Compile Time: {compile_time if use_model_compile else 'N/A'}")
    
    for epoch in tqdm(range(num_epochs), desc=f"Epochs [{dataset_name}]"):
        train_logs = train_dataset.train(
            model = model, 
            dataloader = train_dataloader,
            optimizer = optimizer,
            device = device,
        )
        if use_lr_scheduler:
            scheduler.step()

        eval_logs = eval_dataset.eval(
            model = model,
            dataloader = eval_dataloader,
            device = device,
        )
        if use_wandb:
            wandb.log({
                **train_logs,
                **eval_logs
            })

if __name__ == "__main__":
    main() 