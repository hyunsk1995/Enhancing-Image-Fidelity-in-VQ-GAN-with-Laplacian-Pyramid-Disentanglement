import argparse, os, sys, glob, math, time
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from main import instantiate_from_config, DataModuleFromConfig
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import trange
from kornia.geometry.transform.pyramid import PyrUp

pyrUp = PyrUp()

def save_image(x, path):
    c,h,w = x.shape
    assert c==3
    x = ((x.detach().cpu().numpy().transpose(1,2,0)+1.0)*127.5).clip(0,255).astype(np.uint8)
    Image.fromarray(x).save(path)

# logits : (B, hw, vocab_size)
def AR_modeling(model, idx, cidx, start_i, start_j, qshape, temperature, top_k, hier="top"):
    sample = True

    for i in range(start_i, qshape[2]):
        for j in range(start_j, qshape[3]):
            print("i, j:", i, j)
            cx = torch.cat((cidx, idx), dim=1)
            logits, _ = model.transformer(cx[:,:-1]) # (B, block_size, vocab_size)

            logits = logits[:, -qshape[2]*qshape[3]:, :]
            logits = logits[:, i*qshape[2]+j, :]
            logits /= temperature

            if top_k is not None:
                logits = model.top_k_logits(logits, top_k)
            probs = torch.nn.functional.softmax(logits, dim=-1)

            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            
            idx[:,i*qshape[2]+j] = ix
        
    idx = idx.reshape(qshape[0], qshape[2], qshape[3])

    return idx

@torch.no_grad()
def run_conditional(top_model, bottom_model, dsets, outdir, top_k, temperature, batch_size=1):
    if len(dsets.datasets) > 1:
        split = sorted(dsets.datasets.keys())[0]
        dset = dsets.datasets[split]
    else:
        dset = next(iter(dsets.datasets.values()))
    print("Dataset: ", dset.__class__.__name__)
    for start_idx in trange(0,len(dset)-batch_size+1,batch_size):
        indices = list(range(start_idx, start_idx+batch_size))
        example = default_collate([dset[i] for i in indices])

        x = top_model.get_input("image", example).to(top_model.device)
        for i in range(x.shape[0]):
            save_image(x[i], os.path.join(outdir, "originals",
                                          "{:06}.png".format(indices[i])))

        cond_key = top_model.cond_stage_key
        c = top_model.get_input(cond_key, example).to(top_model.device)

        scale_factor = 1.0
        quant_zt, zt_indices = top_model.encode_to_z(x)
        quant_zb, zb_indices = bottom_model.encode_to_z(x)
        _, ct_indices = top_model.encode_to_c(c)
        _, cb_indices = bottom_model.encode_to_c(c)

        qshape_t = quant_zt.shape
        qshape_b = quant_zb.shape

        _, _, xrec = top_model.first_stage_model.decode(quant_zt, quant_zb)
        for i in range(xrec.shape[0]):
            save_image(xrec[i], os.path.join(outdir, "reconstructions",
                                             "{:06}.png".format(indices[i])))

        # if cond_key == "segmentation":
        #     # get image from segmentation mask
        #     num_classes = c.shape[1]
        #     c = torch.argmax(c, dim=1, keepdim=True)
        #     c = torch.nn.functional.one_hot(c, num_classes=num_classes)
        #     c = c.squeeze(1).permute(0, 3, 1, 2).float()
        #     c = model.cond_stage_model.to_rgb(c)

        t_idx = zt_indices
        b_idx = zb_indices
        half_sample = False
        if half_sample:
            outdir += "-half"
            start_t = t_idx.shape[1]//2
            start_b = b_idx.shape[1]//2
        else:
            start_t = 0
            start_b = 0

        t_idx[:,start_t:] = 0
        # t_idx = t_idx.reshape(qshape_t[0],qshape_t[2],qshape_t[3])
        start_it = start_t//qshape_t[3]
        start_jt = start_t %qshape_t[3]

        b_idx[:,start_b:] = 0
        # b_idx = b_idx.reshape(qshape_b[0],qshape_b[2],qshape_b[3])
        start_ib = start_b//qshape_b[3]
        start_jb = start_b %qshape_b[3]

        t_idx = AR_modeling(top_model, t_idx, ct_indices, start_it, start_jt, qshape_t, temperature, top_k)

        # Condition from the lower level
        cb_indices = torch.cat((cb_indices, t_idx.reshape(qshape_t[0], -1)), dim=1)

        b_idx = AR_modeling(bottom_model, b_idx, cb_indices, start_ib, start_jb, qshape_b, temperature, top_k, hier="bottom")

        x_lf = top_model.decode_to_img(t_idx, qshape_t)
        x_hf = bottom_model.decode_to_img(b_idx, qshape_b)

        xsample = pyrUp(x_lf) + x_hf
        # xsample = top_model.decode_full_img(t_idx, b_idx, qshape_t, qshape_b)
        for i in range(xsample.shape[0]):
            save_image(x_lf[i], os.path.join(outdir, "samples_lf",
                                                "{:06}.png".format(indices[i])))
            save_image(x_hf[i], os.path.join(outdir, "samples_hf",
                                                "{:06}.png".format(indices[i])))
            save_image(xsample[i], os.path.join(outdir, "samples",
                                                "{:06}.png".format(indices[i])))


def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-r",
    #     "--resume",
    #     type=str,
    #     nargs="?",
    #     help="load from logdir or checkpoint in logdir",
    # )
    parser.add_argument(
        "--top",
        type=str,
        nargs="?",
        help="load from top checkpoint in logdir",
    )
    parser.add_argument(
        "--bottom",
        type=str,
        nargs="?",
        help="load from bottom checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="",
    )
    parser.add_argument(
        "--ignore_base_data",
        action="store_true",
        help="Ignore data specification from base configs. Useful if you want "
        "to specify a custom datasets on the command line.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=str,
        help="Where to write outputs to.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Sample from among top-k predictions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    return parser


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    if "ckpt_path" in config.params:
        print("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None
    if "downsample_cond_size" in config.params:
        print("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5
    try:
        if "ckpt_path" in config.params.first_stage_config.params:
            config.params.first_stage_config.params.ckpt_path = None
            print("Deleting the first-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            print("Deleting the cond-stage restore-ckpt path from the config...")
    except:
        pass

    model = instantiate_from_config(config)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Missing Keys in State Dict: {missing}")
        print(f"Unexpected Keys in State Dict: {unexpected}")
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}


def get_data(config):
    # get data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    return data


def load_model_and_dset(top_config, bottom_config, top_ckpt, bottom_ckpt, gpu, eval_mode):
    # get data
    dsets = get_data(top_config)   # calls data.config ...

    # now load the specified checkpoint
    if top_ckpt:
        pl_sd_top = torch.load(top_ckpt, map_location="cpu")
        global_step_top = pl_sd_top["global_step"]
    else:
        pl_sd_top = {"state_dict": None}
        global_step_top = None

    if bottom_ckpt:
        pl_sd_bottom = torch.load(bottom_ckpt, map_location="cpu")
        global_step_bottom = pl_sd_bottom["global_step"]
    else:
        pl_sd_bottom = {"state_dict": None}
        global_step_bottom = None

    top_model = load_model_from_config(top_config.model,
                                   pl_sd_top["state_dict"],
                                   gpu=gpu,
                                   eval_mode=eval_mode)["model"]
    bottom_model = load_model_from_config(bottom_config.model,
                                   pl_sd_bottom["state_dict"],
                                   gpu=gpu,
                                   eval_mode=eval_mode)["model"]
    return dsets, top_model, bottom_model, global_step_top, global_step_bottom


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    top_ckpt = None
    bottom_ckpt = None
    if opt.top:
        if not os.path.exists(opt.top):
            raise ValueError("Cannot find {}".format(opt.top))
        if os.path.isfile(opt.top):
            paths = opt.top.split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
            except ValueError:
                idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            top_ckpt = opt.top
        else:
            assert os.path.isdir(opt.top), opt.top
            logdir = opt.top.rstrip("/")
            top_ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        print(f"logdir:{logdir}")
        top_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt.top_config = top_configs+opt.base

    if opt.bottom:
        if not os.path.exists(opt.bottom):
            raise ValueError("Cannot find {}".format(opt.bottom))
        if os.path.isfile(opt.bottom):
            paths = opt.bottom.split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
            except ValueError:
                idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            bottom_ckpt = opt.bottom
        else:
            assert os.path.isdir(opt.bottom), opt.bottom
            logdir = opt.bottom.rstrip("/")
            bottom_ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        print(f"logdir:{logdir}")
        bottom_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt.bottom_config = bottom_configs+opt.base

    # if opt.config:
    #     if type(opt.config) == str:
    #         opt.base = [opt.config]
    #     else:
    #         opt.base = [opt.base[-1]]

    top_configs = [OmegaConf.load(cfg) for cfg in opt.top_config]
    cli = OmegaConf.from_dotlist(unknown)
    if opt.ignore_base_data:
        for config in top_configs:
            if hasattr(config, "data"): del config["data"]
    top_config = OmegaConf.merge(*top_configs, cli)

    bottom_configs = [OmegaConf.load(cfg) for cfg in opt.bottom_config]
    cli = OmegaConf.from_dotlist(unknown)
    if opt.ignore_base_data:
        for config in bottom_configs:
            if hasattr(config, "data"): del config["data"]
    bottom_config = OmegaConf.merge(*bottom_configs, cli)

    gpu = True
    eval_mode = True
    show_config = False
    if show_config:
        print(OmegaConf.to_container(top_config))
        print(OmegaConf.to_container(bottom_config))

    dsets, top_model, bottom_model, global_step_top, global_step_bottom = load_model_and_dset(top_config, bottom_config, top_ckpt, bottom_ckpt, gpu, eval_mode)
    print(f"Global step: {global_step_top}")

    outdir = os.path.join(opt.outdir, "{:06}_{}_{}".format(global_step_top,
                                                           opt.top_k,
                                                           opt.temperature))
    os.makedirs(outdir, exist_ok=True)
    print("Writing samples to ", outdir)
    for k in ["originals", "reconstructions", "samples", "samples_lf", "samples_hf"]:
        os.makedirs(os.path.join(outdir, k), exist_ok=True)
    run_conditional(top_model, bottom_model, dsets, outdir, opt.top_k, opt.temperature)
