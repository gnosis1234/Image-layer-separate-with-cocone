import pdb

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os 
from PIL import Image

import mmcv
import torch

from mmedit.apis import init_model, inpainting_inference
from mmedit.core import tensor2img


def parse_args():
    parser = argparse.ArgumentParser(description='Inpainting demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('masked_img_path', help='path to input image file')
    parser.add_argument('mask_path', help='path to input mask file')
    parser.add_argument('save_path', help='path to save inpainting result')
    parser.add_argument(
        '--imshow', action='store_true', help='whether show image with opencv')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.device < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    model = init_model(args.config, args.checkpoint, device=device)

    result = inpainting_inference(model, args.masked_img_path, args.mask_path)
    result = tensor2img(result, min_max=(-1, 1))[..., ::-1]
    
    mmcv.mkdir_or_exist(args.save_path)
    # cv2.imwrite(args.save_path, result)
    
    filename = args.masked_img_path.split('/')[-1]
    img = Image.fromarray(result)
    img.save(os.path.join(args.save_path, filename), "PNG")
    # mmcv.imwrite(result, args.save_path)
    if args.imshow:
        mmcv.imshow(result, 'predicted inpainting result')
    

if __name__ == '__main__':
    main()
