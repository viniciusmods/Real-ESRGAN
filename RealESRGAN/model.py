import os
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import cv2
from huggingface_hub import hf_hub_url, cached_download

from .rrdbnet_arch import RRDBNet
from .utils import pad_reflect, split_image_into_overlapping_patches, stich_together, \
                   unpad_image


HF_MODELS = {
    2: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x2.pth',
    ),
    4: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x4.pth',
    ),
    8: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x8.pth',
    ),
}


class RealESRGAN:
    def __init__(self, device, scale=4):
        self.device = device
        self.scale = scale
        self.model = RRDBNet(3, 3, 64, 23, 32, scale).to(device)

    def load_weights(self, model_path, download=True):
        if not os.path.exists(model_path) and download:
            config = HF_MODELS[self.scale]
            cached_download(hf_hub_url(config['repo_id'], config['filename']), cache_dir=os.path.dirname(model_path), force_filename=os.path.basename(model_path))
        
        loadnet = torch.load(model_path)
        self.model.load_state_dict(loadnet.get('params') or loadnet.get('params_ema') or loadnet, strict=True)
        self.model.eval()

    @torch.cuda.amp.autocast()
    def predict(self, lr_image, batch_size=4, patches_size=192, padding=24, pad_size=15):
        lr_image = pad_reflect(np.array(lr_image), pad_size)
        patches, p_shape = split_image_into_overlapping_patches(lr_image, patches_size, padding)
        img = torch.FloatTensor(patches/255).permute((0,3,1,2)).to(self.device).detach()

        with torch.no_grad():
            res = [self.model(img[i:i+batch_size]) for i in range(0, img.shape[0], batch_size)]
        
        sr_image = torch.cat(res, 0).permute((0,2,3,1)).clamp_(0, 1).cpu().numpy()
        sr_img = unpad_image((stich_together(sr_image, tuple(np.multiply(p_shape[0:2], self.scale)) + (3,), tuple(np.multiply(lr_image.shape[0:2], self.scale)) + (3,), padding * self.scale)*255).astype(np.uint8), pad_size*self.scale)
        
        return Image.fromarray(sr_img)
