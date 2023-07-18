import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init

import cv2
import numpy as np


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


class VQVAE2Loss(nn.Module):
    def __init__(self, codebook_weight_t=1.0, codebook_weight_b=1.0):
        super().__init__()
        self.codebook_weight_t = codebook_weight_t
        self.codebook_weight_b = codebook_weight_b

    def forward(self, codebook_loss_t, codebook_loss_b, inputs, lf_recon, hf_recon, reconstructions, split="train"):
        lf, hf = disentangle(inputs)
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        lf_loss = torch.abs(lf.contiguous(), lf_recon.contiguous())
        hf_loss = torch.abs(hf.contiguous(), hf_recon.contiguous())

        loss = rec_loss.mean() + lf_loss.mean() + hf_loss.mean() + self.codebook_weight_t * codebook_loss_t.mean() + self.codebook_weight_b * codebook_loss_b.mean()

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss_top".format(split): codebook_loss_t.detach().mean(),
                "{}/quant_loss_bottom".format(split): codebook_loss_b.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/lf_loss".format(split): lf_loss.detach().mean(),
                "{}/hf_loss".format(split): hf_loss.detach().mean(),
                }
        return loss, log
    
class MultiStageTransformerLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, codebook_loss, inputs, reconstructions, split="train"):
        return
    
def disentangle(img):
    print(img.shape)
    assert img.shape == [256,256,3]
    img = cv2.GaussianBlur(img, (3,3), 0)
    Downimg = cv2.pyrDown(img)
    DownUp = cv2.pyrUp(Downimg)

    imgarray = np.array(img, dtype=int)
    DownUparray = np.array(DownUp, dtype=int)
    Laplacian = imgarray - DownUparray

    return Downimg, Laplacian
