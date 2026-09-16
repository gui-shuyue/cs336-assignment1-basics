import torch
from torch import Tensor
from collections.abc import Iterable


def _logsumexp(x: Tensor, keepdim=False):
    m = torch.max(x, dim=-1, keepdim=True).values
    result = m + torch.log(Tensor.sum(torch.exp(x - m), dim=-1, keepdim=True))
    if not keepdim:
        result = result.squeeze(-1)
    return result


def softmax(x: Tensor, dim: int) -> Tensor:
    maximum = x.max(dim=dim, keepdim=True).values
    shifted = x - maximum
    exponentials = torch.exp(shifted)
    denominator = exponentials.sum(dim=dim, keepdim=True)
    return exponentials / denominator


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """
    Computes the cross-entropy loss between the predicted inputs and the target labels.

    Args:
        inputs (Tensor): The predicted logits from the model (before softmax).
        targets (Tensor): The true labels (as class indices).

    Returns:
        Tensor: The computed cross-entropy loss.
    """
    log_denominators = _logsumexp(inputs)
    target_indices = targets.unsqueeze(-1)
    target_logits = torch.gather(inputs, dim=-1, index=target_indices).squeeze(-1)
    loses = log_denominators - target_logits
    return loses.mean()


@torch.no_grad()
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    """Clip the combined L2 norm of all existing parameter gradients in-place."""
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return

    squared_norm = torch.zeros((), device=gradients[0].device, dtype=torch.float32)
    for gradient in gradients:
        values = gradient.coalesce().values() if gradient.is_sparse else gradient
        squared_norm += values.detach().float().pow(2).sum().to(squared_norm.device)

    total_norm = squared_norm.sqrt()
    scale = torch.clamp(max_l2_norm / (total_norm + eps), max=1.0)
    for gradient in gradients:
        gradient.mul_(scale.to(device=gradient.device, dtype=gradient.dtype))
