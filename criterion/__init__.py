import torch.nn as nn
import torch

class CrossEntropyLoss(nn.Module):
    def __init__(self, pad_id):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            reduction="mean",
            ignore_index=pad_id,
        )
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        
        return self.cross_entropy_loss(
            logits.contiguous().view(-1, logits.size(-1)),
            targets.contiguous().view(-1),
        )
