import torch
from torch import Tensor

def _logsumexp(x: Tensor, keepdim=False):
    m = torch.max(x, dim=-1, keepdim=True).values
    result = m+torch.log(Tensor.sum(torch.exp(x-m), dim=-1, keepdim=True))
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