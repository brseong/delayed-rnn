import torch
import torch.nn as nn
import torch.nn.functional as F

from jaxtyping import Float
from torch.nn.functional import softplus
from utils.config import Config


class LearnableDelayRNNBackbone(nn.Module):
    """
    Shared backbone for LearnableDelayRNN-based models.
    Contains: afferent, lateral, efferent weights, tau, scale_exponent,
    credit matrix computation, and gather-based step_fast.
    
    Subclasses only need to implement forward().
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, max_delay: int, config: Config):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        assert max_delay >= 1, "max_delay must be at least 1"
        self.max_delay = max_delay
        self.device = config.device
        self.config = config

        # Core weights
        self.afferent = nn.Linear(input_size, hidden_size)
        self.lateral = nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty(hidden_size, hidden_size))  # hidden_out, hidden_in
        )
        self.efferent = nn.Linear(hidden_size, output_size)

        self.tau = nn.Parameter(max_delay * torch.rand_like(self.lateral) + 1)
        self.scale_exponent = nn.Parameter(torch.zeros(hidden_size, device=config.device))

    # ------------------------------------------------------------------
    # Credit matrix
    # ------------------------------------------------------------------
    @staticmethod
    @torch.jit.script
    def calc_credit_matrix_jit(tau_clipped: torch.Tensor, max_delay: int, hidden_size: int, scale_exponent: torch.Tensor) -> torch.Tensor:
        credit_matrix = torch.arange(max_delay + 1, out=tau_clipped.new_empty(max_delay + 1))  # (max_delay+1,)
        credit_matrix = credit_matrix[:, None, None]
        distance = 1.0 + torch.abs(credit_matrix - tau_clipped)

        # ### Credit matrix with integration:
        # raw_credit = distance.pow(-softplus(scale_exponent[None, :, None]))
        # credit_matrix = raw_credit / (raw_credit.sum(dim=0, keepdim=True))  # Normalize to sum to 1 across delay dimension
        # return credit_matrix
        
    
        ### Credit matrix with Gumbel-Softmax sampling (alternative):
        raw_credit = distance.pow(-scale_exponent)
        logits = torch.log(raw_credit + 1e-8)
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
        perturbed_logits = logits + gumbel_noise
        differentiable = F.softmax(perturbed_logits, dim=0) # Forward and backward pass goes through this path
        
        max_idx = perturbed_logits.argmax(dim=0, keepdim=True)
        nondifferentiable = torch.zeros_like(logits).scatter_(0, max_idx, 1.0)
        
        # Straight-Through Estimator trick
        # Return equals a spike in the tau_clipped position,
        # but gradients flow through the differentiable soft assignment, enabling learning of tau.
        return differentiable + (nondifferentiable - differentiable).detach()
    

    def calc_credit_matrix(self) -> Float[torch.Tensor, "max_delay+1 hidden_out hidden_in"]:
        return LearnableDelayRNNBackbone.calc_credit_matrix_jit(
            torch.clamp(self.tau, 1, self.max_delay)[None, ...],
            self.max_delay,
            self.hidden_size,
            self.scale_exponent)

    # ------------------------------------------------------------------
    # Pre-computation helper (credit_matrix → W_rev)
    # ------------------------------------------------------------------
    def precompute_W_rev(self) -> torch.Tensor:
        """Returns the reversed weight matrix ready for step_fast."""
        credit_matrix = self.calc_credit_matrix()
        W = credit_matrix[1:] * self.lateral[None, :, :]  # (max_delay, hidden, hidden)
        W_rev = W.flip(0)  # Pre-flip to match the ordering of past records
        return W_rev

    # ------------------------------------------------------------------
    # Gather-based O(1)-write step function
    # ------------------------------------------------------------------
    def step_fast(self, x_t, history, ptr, W_rev):
        """
        Gather (pull) step function with O(1) memory write.
        x_t:      (batch, input_size)
        history:  (max_delay, batch, hidden_size) - past h_to_delay records
        W_rev:    (max_delay, hidden_size, hidden_size) - reversed weight matrix
        Returns:  (history, ptr, y_t)
        """
        # 1. Lightly rotate weight matrix W by pointer position
        W_aligned = torch.roll(W_rev, shifts=ptr, dims=0)

        # 2. Multiply past history and weights, then aggregate delayed signals arriving at current step (Gather)
        h_delayed = torch.einsum('dhi,dbi->bh', W_aligned, history)

        # 3. Compute new hidden state
        h_to_delay = torch.tanh(self.afferent(x_t) + h_delayed)

        # 4. Update buffer (use torch.where to avoid autograd in-place errors)
        mask = (torch.arange(self.max_delay, device=x_t.device) == ptr).view(-1, 1, 1)
        history = torch.where(mask, h_to_delay.unsqueeze(0), history)

        # 5. Move pointer and compute output
        ptr = (ptr + 1) % self.max_delay
        y_t = self.efferent(h_delayed)

        return history, ptr, y_t

    # ------------------------------------------------------------------
    # Buffer initialisation helper
    # ------------------------------------------------------------------
    def init_history(self, batch_size: int, *, ref_tensor: torch.Tensor | None = None) -> torch.Tensor:
        """Return a zero-initialised history buffer (max_delay, batch, hidden)."""
        if ref_tensor is not None:
            return ref_tensor.new_zeros(self.max_delay, batch_size, self.hidden_size)
        return torch.zeros(self.max_delay, batch_size, self.hidden_size, device=self.device)
