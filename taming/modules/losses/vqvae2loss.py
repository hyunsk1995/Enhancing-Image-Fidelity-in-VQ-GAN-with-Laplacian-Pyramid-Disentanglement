import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


class VQVAE2Loss(nn.Module):
    def __init__(self, codebook_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(self, codebook_loss, inputs, reconstructions, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        loss = rec_loss.mean() + self.codebook_weight * codebook_loss.mean()

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                }
        return loss, log
    
class MultiStageTransformerLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, codebook_loss, inputs, reconstructions, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        loss = rec_loss.mean() + self.codebook_weight * codebook_loss.mean()

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                }
        return loss, log