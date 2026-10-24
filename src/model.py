import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn


def build_model(arch: str = "unet", encoder: str = "resnet34", weights: str | None = "imagenet") -> nn.Module:
    """smp model with an ImageNet encoder; the RGB stem is collapsed to 1 channel by smp."""
    return smp.create_model(arch, encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=1)


class BCEDiceLoss(nn.Module):
    """bce_weight * BCE + (1 - bce_weight) * soft Dice (per image). Dice is insensitive to the ~99%
    background pixels, so the rare filament class is not swamped; BCE keeps gradients well-behaved."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight, self.smooth = bce_weight, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum((1, 2, 3))
        dice = 1 - (2 * inter + self.smooth) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + self.smooth)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice.mean()


def model_size(model: nn.Module) -> dict:
    """Parameter count and in-memory size of the weights."""
    n = sum(p.numel() for p in model.parameters())
    nbytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    return {"params_M": n / 1e6, "size_MB": nbytes / 2 ** 20}
