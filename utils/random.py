import torch

def clipped_gamma_sample(x:torch.Tensor, max_think_steps:int, alpha=2.0, scale=0.3) -> torch.Tensor:
    """클리핑된 감마 분포에서 샘플링하는 함수"""
    # 감마 분포 정의 (shape=2.0, scale=0.15)
    device = x.device
    gamma_dist = torch.distributions.Gamma(
        torch.tensor([alpha], device=device), 
        torch.tensor([scale], device=device)
    )
    samples = gamma_dist.sample((x.shape[0],)).to(device)
    # 클리핑: 최대 max_think_steps를 넘지 않도록 제한
    # samples = torch.clamp(samples, max=max_think_steps)
    samples.clamp_(max=max_think_steps)
    return samples.squeeze(-1)