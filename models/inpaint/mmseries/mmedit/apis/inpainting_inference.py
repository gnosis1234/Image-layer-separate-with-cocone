# Copyright (c) OpenMMLab. All rights reserved.
import pdb 

import os
import torch
from mmcv.parallel import collate, scatter

from mmedit.datasets.pipelines import Compose
from torchvision.utils import save_image

def inpainting_inference(model, imgs, masks):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        masked_img (str): File path of image with mask.
        mask (str): Mask file path.

    Returns:
        Tensor: The predicted inpainting result.
    """
    # masked_imgs = masked_imgs.cpu().numpy()
    # masks = masks.cpu().numpy()
    device = next(model.parameters()).device  # model device

    infer_pipeline = [
        dict(type='LoadImageFromFile', key='masked_img', channel_order='bgr'),
        dict(type='LoadMask', mask_mode='mi', mask_config=dict()),
        # dict(type='Pad', keys=['masked_img', 'mask'], mode='reflect'),
        dict(
            type='Normalize',
            keys=['masked_img'],
            mean=[127.5] * 3,
            std=[127.5] * 3,
            to_rgb=True),
        dict(type='GetMaskedImage', img_name='masked_img'),
        dict(type='FramesToTensor', keys=['masked_img', 'mask']),
        dict(
            type='Collect',
            keys=['masked_img', 'mask'])        
    ]


    masked_imgs = imgs * (1. - masks)
    data = {'masked_img': masked_imgs, 'mask': masks}
    # forward the model
    with torch.no_grad():
        result = model(test_mode=True, **data)
    
    return result['fake_img']
