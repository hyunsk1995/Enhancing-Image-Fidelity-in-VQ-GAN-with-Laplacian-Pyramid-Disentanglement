import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init

import cv2
from kornia.geometry.transform.pyramid import PyrUp, PyrDown
from torchvision.transforms import GaussianBlur

import numpy as np

pyrUp = PyrUp()
pyrDown = PyrDown()
gaussianBlur = GaussianBlur((3,3), 1.5)

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
        lf_loss = torch.abs(lf.clone().detach() - lf_recon)
        hf_loss = torch.abs(hf.clone().detach() - hf_recon
                            
        loss = rec_loss.mean() + lf_loss.mean() + hf_loss.mean() + self.codebook_weight_t * codebook_loss_t.mean() + self.codebook_weight_b * codebook_loss_b.mean()
        # loss = hf_loss.mean()

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
    assert img[0].shape == (3, 256, 256)    
    img = gaussianBlur(img)
    
    Downimg = pyrDown(img)
    DownUp = pyrUp(Downimg)

    # imgarray = np.array(img.detach().cpu(), dtype=float)
    # DownUparray = np.array(DownUp.detach().cpu(), dtype=float)
    # Laplacian = imgarray - DownUparray
    Laplacian = img - DownUp

    return Downimg, Laplacian
