import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn

from main import instantiate_from_config
from taming.modules.util import SOSProvider

from kornia.geometry.transform.pyramid import PyrUp, PyrDown

pyrUp = PyrUp()
pyrDown = PyrDown()


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class MultiStageTransformer(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 downsample_cond_size=-1,
                 pkeep=1.0,
                 sos_token=0,
                 unconditional=True,
                 num_stages=2,
                 ):
        super().__init__()
        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.num_stages = num_stages
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        if permuter_config is None:
            permuter_config = {"target": "taming.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        # self.transformer = instantiate_from_config(config=transformer_config)
        self.transformer = nn.ModuleList()
        for stage in range(num_stages):
            self.transformer.append(instantiate_from_config(config=transformer_config[stage]))

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__" or self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key
            self.cond_stage_model = SOSProvider(self.sos_token)
        else:
            model = instantiate_from_config(config)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model

    def forward(self, x, c):
        # one step to produce the logits
        p_indices = None
        logits = []
        targets = []

        quant_z, info = self.encode_to_z(x)

        for i in range(self.num_stages):
            hier = (self.num_stages-1) -i
            _, c_indices = self.encode_to_c(c)

            z_indices = info[hier][2].view(quant_z[hier].shape[0], -1)
            z_indices = self.permuter(z_indices)

            if self.training and self.pkeep < 1.0:
                mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                            device=z_indices.device))
                mask = mask.round().to(dtype=torch.int64)
                r_indices = torch.randint_like(z_indices, self.transformer[hier].config.vocab_size)
                a_indices = mask*z_indices+(1-mask)*r_indices
            else:
                a_indices = z_indices

            if p_indices is not None:
                a_indices = torch.cat((p_indices, a_indices), dim=1)
            cz_indices = torch.cat((c_indices, a_indices), dim=1)

            # target includes all sequence elements (no need to handle first one
            # differently because we are conditioning)
            target = z_indices
            # make the prediction
            logit, _, feature = self.transformer[hier](cz_indices[:, :-1])

            if p_indices is not None:
                logit = logit[:,p_indices.shape[1]:]
            # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
            logit = logit[:, c_indices.shape[1]-1:]
            logits.append(logit)
            targets.append(target)
            p_indices = z_indices

        return logits, targets

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out
    
    @torch.no_grad()
    def sample(self, x, c, steps, hier, prev, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        if prev is not None:
            x = torch.cat((prev, x),dim=1)
        x = torch.cat((c,x),dim=1)
        block_size = self.transformer[hier].get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape)==2
            noise_shape = (x.shape[0], steps-1)
            #noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
            x = torch.cat((x,noise),dim=1)
            logits, _, feature = self.transformer[hier](x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            if prev is not None:
                x = ix[:, prev.shape[1]:]
            x = x[:, c.shape[1]-1:]
        else:
            for k in range(steps):
                callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _, feature = self.transformer[hier](x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            # cut off conditioning
            if prev is not None:
                x = x[:, prev.shape[1]:]
            x = x[:, c.shape[1]:]
        return x, feature

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        return quant_z, info

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape, hier):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize[hier].get_codebook_entry(
                index.reshape(-1), shape=bhwc)

        quant_z = self.first_stage_model.post_quant_conv[hier](quant_z)
        x = self.first_stage_model.decoder[hier](quant_z)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()

        N = 4
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)

        # quant_z, info = self.encode_to_z(x)
        quant_z, _, info = self.first_stage_model.encode(x)
        p_indices = None
        
        for i in range(self.num_stages):
            hier = (self.num_stages-1) - i
            quant_c, c_indices = self.encode_to_c(c)

            z_indices = info[hier][-1].view(quant_z[hier].shape[0], -1)
            z_indices = self.permuter(z_indices)

            # create a "half"" sample
            z_start_indices = z_indices[:,:z_indices.shape[1]//2]
            # z_start_indices = z_indices[:, :0]
            
            index_sample, feature = self.sample(z_start_indices, c_indices,
                                       steps=z_indices.shape[1]-z_start_indices.shape[1],
                                       prev = p_indices,
                                       hier=hier,
                                       temperature=temperature if temperature is not None else 1.0,
                                       sample=True,
                                       top_k=top_k if top_k is not None else 100,
                                       callback=callback if callback is not None else lambda k: None)
            x_sample = self.decode_to_img(index_sample, quant_z[hier].shape, hier)

            x_rec = self.decode_to_img(z_indices, quant_z[hier].shape, hier)

            if i > 0:
                full_rec = pyrUp(full_rec) + x_rec
            else:
                full_rec = x_rec


            log["recon_stage"+str(i+1)] = x_rec
            log["sample_stage"+str(i+1)] = x_sample
            p_indices = z_indices

        # log["samples_half"] = x_sample
        # log["samples_nopix"] = x_sample_nopix
        # log["samples_det"] = x_sample_det
        log["recon_full"] = full_rec
        log["inputs"] = x

        return log


        # create a "half"" sample
        # z_start_indices = z_indices[:,:z_indices.shape[1]//2]
        # index_sample = self.sample(z_start_indices, c_indices,
        #                            steps=z_indices.shape[1]-z_start_indices.shape[1],
        #                            temperature=temperature if temperature is not None else 1.0,
        #                            sample=True,
        #                            top_k=top_k if top_k is not None else 100,
        #                            callback=callback if callback is not None else lambda k: None)
        # x_sample = self.decode_to_img(index_sample, quant_z.shape)

        # sample
        # z_start_indices = z_indices[:, :0]
        # index_sample = self.sample(z_start_indices, c_indices,
        #                            steps=z_indices.shape[1],
        #                            temperature=temperature if temperature is not None else 1.0,
        #                            sample=True,
        #                            top_k=top_k if top_k is not None else 100,
        #                            callback=callback if callback is not None else lambda k: None)
        # x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)

        # det sample
        # z_start_indices = z_indices[:, :0]
        # index_sample = self.sample(z_start_indices, c_indices,
        #                            steps=z_indices.shape[1],
        #                            sample=False,
        #                            callback=callback if callback is not None else lambda k: None)
        # x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # reconstruction
        # x_rec = self.decode_to_img(z_indices, quant_z.shape)

        # log["inputs"] = x
        # log["reconstructions"] = x_rec

        # if self.cond_stage_key in ["objects_bbox", "objects_center_points"]:
        #     figure_size = (x_rec.shape[2], x_rec.shape[3])
        #     dataset = kwargs["pl_module"].trainer.datamodule.datasets["validation"]
        #     label_for_category_no = dataset.get_textual_label_for_category_no
        #     plotter = dataset.conditional_builders[self.cond_stage_key].plot
        #     log["conditioning"] = torch.zeros_like(log["reconstructions"])
        #     for i in range(quant_c.shape[0]):
        #         log["conditioning"][i] = plotter(quant_c[i], label_for_category_no, figure_size)
        #     log["conditioning_rec"] = log["conditioning"]
        # elif self.cond_stage_key != "image":
        #     cond_rec = self.cond_stage_model.decode(quant_c)
        #     if self.cond_stage_key == "segmentation":
        #         # get image from segmentation mask
        #         num_classes = cond_rec.shape[1]

        #         c = torch.argmax(c, dim=1, keepdim=True)
        #         c = F.one_hot(c, num_classes=num_classes)
        #         c = c.squeeze(1).permute(0, 3, 1, 2).float()
        #         c = self.cond_stage_model.to_rgb(c)

        #         cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
        #         cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
        #         cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
        #         cond_rec = self.cond_stage_model.to_rgb(cond_rec)
        #     log["conditioning_rec"] = cond_rec
        #     log["conditioning"] = c

        # return log

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        logits, targets = self(x, c)
        loss = []
        for logit, target in zip(logits, targets):
            loss.append(F.cross_entropy(logit.reshape(-1, logit.size(-1)), target.reshape(-1)))
        
        return loss #sum(loss)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        for i, l in enumerate(loss):
            self.log("train/loss/stage_{}".format(i+1), l, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return sum(loss)

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        for i, l in enumerate(loss):
            self.log("val/loss/stage_{}".format(i+1), l, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return sum(loss)

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for stage in range(self.num_stages):
            for mn, m in self.transformer[stage].named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for stage in range(self.num_stages) for pn, p in self.transformer[stage].named_parameters()}
        param_list = list()
        for stage in range(self.num_stages):
            param_list += self.transformer[stage].named_parameters()
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        # for key in param_list:
        #     print(key[0])

        # for key in param_dict:
        #     print(key)
        # create the pytorch optimizer object
        optim_groups = []

        for key in param_list:
            if key[0] in decay:
                optim_groups.append({"params": key[1], "weight_decay": 0.01})
            else:
                optim_groups.append({"params": key[1], "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer
