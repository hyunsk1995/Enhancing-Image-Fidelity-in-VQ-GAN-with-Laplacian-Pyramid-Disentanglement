import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

import cv2
import torch.nn as nn

from kornia.geometry.transform.pyramid import PyrUp, PyrDown
from torchvision.transforms import GaussianBlur

pyrUp = PyrUp()
pyrDown = PyrDown()
gaussianBlur = GaussianBlur((3,3), 1.5)

class HierarchicalVQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 num_stages=2,
                 ):
        super().__init__()

        num_latent_layers = ddconfig["num_latent_layers"]
        n_res_block = ddconfig["num_res_blocks"]
        n_res_channel = ddconfig["residual_units"]
        in_channel = ddconfig["in_channels"]
        channel = ddconfig["ch"]
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.quant_conv = nn.ModuleList()
        self.quantize = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.num_stages = num_stages

        self.encoder.append(Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4))

        for i in range(self.num_stages):
            if i > 0:
                self.encoder.append(Encoder(channel, channel, n_res_block, n_res_channel, stride=2))
            self.quant_conv.append(torch.nn.Conv2d(channel, embed_dim, 1))
            self.quantize.append(VectorQuantizer(embed_dim, n_embed[i]))
            self.decoder.append(Decoder(embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=4))

        self.loss = instantiate_from_config(lossconfig)        
        # self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        self.image_key = image_key
        
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        xrec, dec = self.decode(quant)

        return xrec, dec, diff

    def encode(self, input):
        prev = input
        enc = []
        quant = []
        diff = []
        id = []

        for i in range(self.num_stages):
            _enc = self.encoder[i](prev)
            _quant = self.quant_conv[i](_enc).permute(0, 2, 3, 1)
            _quant, _diff, _id = self.quantize[i](_quant)
            _quant = _quant.permute(0, 3, 1, 2)
            _diff = _diff.unsqueeze(0)

            enc.append(_enc)
            quant.append(_quant)
            diff.append(_diff)
            id.append(_id)
           
            prev = enc[i]

        return quant, diff, id
    
    def decode(self, quant):
        dec = []
        for i in range(self.num_stages):
            j = (self.num_stages-1) - i
            dec.append(self.decoder[j](quant[j]))

            if i > 0:
                xrec = pyrUp(xrec) + dec[i]
            else:
                xrec = dec[i]
        return xrec, dec

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, dec, diff = self(x)
        
        # autoencode
        aeloss, log_dict_ae = self.loss(diff, x, dec, xrec, split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, dec, diff = self(x)
        
        # autoencode
        aeloss, log_dict_ae = self.loss(diff, x, dec, xrec, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_list = list()
        for i in range(self.num_stages):
            opt_list += list(self.encoder[i].parameters())
            opt_list += list(self.decoder[i].parameters())
            opt_list += list(self.quantize[i].parameters())
            opt_list += list(self.quant_conv[i].parameters())

        opt_ae = torch.optim.Adam(opt_list,
                                  lr=lr, betas=(0.5, 0.9))
        
        return [opt_ae], []

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, dec, _ = self(x)

        dis = self.disentangle(x, self.num_stages)

        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        for stage in range(self.num_stages):
            log["input_stage{}".format(stage+1)] = dis[stage]
            log["recon_stage{}".format(stage+1)] = dec[stage]
        
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
    
    def disentangle(self, img, num_stage):
        assert img[0].shape == (3, 256, 256)    
        img = gaussianBlur(img)
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
