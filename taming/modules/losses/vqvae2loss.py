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
    def __init__(self, codebook_weight=[1.0, 1.0]):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(self, codebook_loss, inputs, dec, xrec, split="train"):
        num_stage = len(dec)
        disentangled = disentangle(inputs, num_stage)
        disentangle_loss = []
        rec_loss = torch.abs(inputs.contiguous() - xrec.contiguous())
        loss = rec_loss.mean()
        log_list = dict()
        
        for i in range(num_stage):
            disentangle_loss.append(torch.abs(disentangled[i] - dec[i]))
            loss += disentangle_loss[i].mean()
            loss += codebook_loss[i].mean() * self.codebook_weight[i]
            log_list["{}/stage{}_loss".format(split, i+1)] = disentangle_loss[i].detach().mean()
        

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss_top".format(split): codebook_loss[0].detach().mean(),
                "{}/quant_loss_bottom".format(split): codebook_loss[1].detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean()
                }
        
        log.update(log_list)
        return loss, log


def disentangle(img, num_stage):
    assert img[0].shape == (3, 256, 256)    
    # img = gaussianBlur(img)
    disentangled = []
    prev = img

    for _ in range(num_stage-1):
        Downimg = pyrDown(prev)
        DownUp = pyrUp(Downimg)
        Laplacian = prev - DownUp
        prev = Downimg
        disentangled.append(Laplacian)
    
    disentangled.append(Downimg)
    disentangled.reverse()

    return disentangled
