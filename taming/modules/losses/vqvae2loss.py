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

            # if i == 0:
            #     loss += disentangle_loss[i].mean()
            loss += codebook_loss[i].mean() * self.codebook_weight[i]
            log_list["{}/stage{}_loss".format(split, i+1)] = disentangle_loss[i].detach().mean()
            log_list["{}/quant_loss_stage{}".format(split, i+1)] = codebook_loss[i].detach().mean()
        

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                # "{}/quant_loss_top".format(split): codebook_loss[0].detach().mean(),
                # "{}/quant_loss_bottom".format(split): codebook_loss[1].detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean()
                }
        
        log.update(log_list)
        return loss, log


def disentangle(img, num_stage):
    assert img[0].shape == (3, 256, 256)    
    img = gaussianBlur(img)
    disentangled = []
    prev = img

    if num_stage == 1:
        return img
    
    for _ in range(num_stage-1):
        Downimg = pyrDown(prev)
        DownUp = pyrUp(Downimg)
        Laplacian = prev - DownUp
        prev = Downimg
        disentangled.append(Laplacian)
    
    disentangled.append(Downimg)
    disentangled.reverse()

    return disentangled

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=[1.0, 1.0], pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, dec, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        num_stage = len(dec)
        disentangled = disentangle(inputs, num_stage)
        disentangle_loss = []

        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)
        loss = nll_loss

        log_list = dict()

        for i in range(num_stage):
            disentangle_loss.append(torch.abs(disentangled[i] - dec[i]))

            if i == 0:
                loss += disentangle_loss[i].mean()
            loss += codebook_loss[i].mean() * self.codebook_weight[i]

            log_list["{}/stage{}_loss".format(split, i+1)] = disentangle_loss[i].detach().mean()
            log_list["{}/quant_loss_stage{}".format(split, i+1)] = codebook_loss[i].detach().mean()

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss += d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                #    "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            
            log.update(log_list)
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
