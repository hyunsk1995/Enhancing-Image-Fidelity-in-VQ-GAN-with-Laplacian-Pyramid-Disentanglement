import io
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import yaml
import torch
from omegaconf import OmegaConf
from taming.models.vqgan2 import HierarchicalVQModel
from kornia.geometry.transform.pyramid import PyrUp

pyrUp = PyrUp()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    model = HierarchicalVQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  quant_t, quant_b, diff_t, diff_b, id_t, id_b = model.encode(x)
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {quant_t.shape[2:]}")
  xrec, rec_t, rec_b = model.decode(quant_t, quant_b)
  return rec_t, rec_b, xrec

log = "2023-08-02T13-07-38"
config256 = load_config("logs/{}_ffhq256_vqgan/configs/{}-project.yaml".format(log, log))
model256 = load_vqgan(config256, "logs/{}_ffhq256_vqgan/testtube/version_0/checkpoints/epoch=199.ckpt".format(log)).to(DEVICE)
font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf", 22)

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))


def preprocess(img, target_image_size=256, map_dalle=True):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img

def stack_reconstructions(input, x0, xt, xb, titles=[]):
  w, h = input.size[0], input.size[1]
  img = Image.new("RGB", (4*w, h))
  img.paste(input, (0,0))
  img.paste(x0, (1*w,0))
  img.paste(xt, (2*w,0))
  img.paste(xb, (3*w,0))
  for i, title in enumerate(titles):
    ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255), font=font) # coordinates, text, color, font
  img.save("test.jpg")
  return img

titles=["Input", "VQVAE2(256)", "VQVAE2(top)", "VQVAE2(bottom)"]

def reconstruction_pipeline(url, size=256):
  x_vqgan = preprocess(download_image(url), target_image_size=size, map_dalle=False)
  x_vqgan = x_vqgan.to(DEVICE)
  
  print(f"input is of size: {x_vqgan.shape}")
  xt, xb, x0 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model256)
  xt = pyrUp(xt)
  img = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[0])),
                              custom_to_pil(x0[0]), 
                              custom_to_pil(xt[0]), 
                              custom_to_pil(xb[0]), 
                              titles=titles)
  return img

reconstruction_pipeline(url='https://heibox.uni-heidelberg.de/f/7bb608381aae4539ba7a/?dl=1', size=256)