import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer


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
                 ):
        super().__init__()

        num_latent_layers = ddconfig["num_latent_layers"]
        n_res_block = ddconfig["num_res_blocks"]
        n_res_channel = ddconfig["residual_units"]
        in_channel = ddconfig["in_channels"]
        channel = ddconfig["ch"]
        beta = ddconfig["beta"]

        self.encoder_t = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.encoder_b = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)

        self.quant_conv_t = torch.nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = VectorQuantizer(n_embed, embed_dim)

        self.decoder_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )

        self.quant_conv_b = torch.nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = VectorQuantizer(embed_dim, n_embed)
        self.upsample_t = torch.nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )

        self.decoder = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

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
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec
    
    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec

    # def get_input(self, batch, k):
    #     x = batch[k]
    #     if len(x.shape) == 3:
    #         x = x[..., None]
    #     x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
    #     return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        # if optimizer_idx == 1:
        #     # discriminator
        #     discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
        #                                     last_layer=self.get_last_layer(), split="train")
        #     self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        #     self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        #     return discloss

    # def validation_step(self, batch, batch_idx):
    #     x = self.get_input(batch, self.image_key)
    #     xrec, qloss = self(x)
    #     aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
    #                                         last_layer=self.get_last_layer(), split="val")

    #     discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
    #                                         last_layer=self.get_last_layer(), split="val")
    #     rec_loss = log_dict_ae["val/rec_loss"]
    #     self.log("val/rec_loss", rec_loss,
    #                prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
    #     self.log("val/aeloss", aeloss,
    #                prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
    #     self.log_dict(log_dict_ae)
    #     self.log_dict(log_dict_disc)
    #     return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder_t.parameters())+
                                  list(self.encoder_b.parameters())+
                                  list(self.decoder_t.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize_t.parameters())+
                                  list(self.quantize_b.parameters())+
                                  list(self.quant_conv_t.parameters())+
                                  list(self.quant_conv_b.parameters()),
                                #   list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        # opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
        #                             lr=lr, betas=(0.5, 0.9))
        return [opt_ae], []

    # def get_last_layer(self):
    #     return self.decoder.conv_out.weight

    # def log_images(self, batch, **kwargs):
    #     log = dict()
    #     x = self.get_input(batch, self.image_key)
    #     x = x.to(self.device)
    #     xrec, _ = self(x)
    #     if x.shape[1] > 3:
    #         # colorize with random projection
    #         assert xrec.shape[1] > 3
    #         x = self.to_rgb(x)
    #         xrec = self.to_rgb(xrec)
    #     log["inputs"] = x
    #     log["reconstructions"] = xrec
    #     return log

    # def to_rgb(self, x):
    #     assert self.image_key == "segmentation"
    #     if not hasattr(self, "colorize"):
    #         self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
    #     x = F.conv2d(x, weight=self.colorize)
    #     x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
    #     return x