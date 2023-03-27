# Copyright (c) OpenMMLab. All rights reserved.
import pdb 

import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor



def init_segmentor(config, checkpoint=None, device='cpu'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Conf∂big`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    # print("===================================")
    # print(model.CLASSES,
    #     model.PALETTE)
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_segmentor(model, imgs, crf=False):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = []
    imgs = imgs if isinstance(imgs, list) else [imgs]
    
    for img in imgs:
        img_data = dict(img=img)
        img_data = test_pipeline(img_data)
        data.append(img_data)
    data = collate(data, samples_per_gpu=len(imgs))
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, crf=crf, **data)
    return result


def show_result_pyplot(model,
                       img,
                       result,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=0.5,
                       title='',
                       block=True,
                       out_file=None):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        block (bool): Whether to block the pyplot figure.
            Default is True.
        out_file (str or None): The path to write the image.
            Default: None.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.title(title)
    plt.tight_layout()
    plt.show(block=block)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

import numpy as np
import os.path as ops
from PIL import Image, ImageFilter
from scipy import ndimage, misc
import cv2
# image = image.filter(ImageFilter.ModeFilter(size=13))
# result = ndimage.median_filter(ascent, size=20)


def layer_separate_imwrite(img, result, class_name=None, out_file=None, fname='mode'):
    alpha_zero = np.zeros((np.shape(img[1]))).astype('bool')
    if len(img) == 2:
        alpha_zero = img[1] < 30
        img = img[0]
    img = mmcv.imread(img)
    img = img.copy()
    seg = result[0]

    segmentation = dict()
    for label, name in enumerate(class_name):
        erosion = False
        if name in ['eyeball_H', 'eyelid1_B']:
            erosion = True
        segmentation[name] = post_processing(fname, label, seg, erosion=erosion) #(label == seg)
        # if name in ['eyeball_H', 'eyelid1_B']:
        #     segmentation[name] = post_processing('erosion', label, segmentation[name])
    for label, name in enumerate(class_name):
        if label == 0: continue

        label_map = segmentation[name]
        mask_idx = (label_map == 0)
        mask = np.ones((seg.shape[0], seg.shape[1]))
        if name == 'eyelid1_B':
            # idx = segmentation['eyeball_H'].astype(bool) | segmentation['eyelid_F'].astype(bool)
            # mask[idx] = 1
            idx = segmentation['eyelid1_B'].astype(bool) == True
            mask[idx] = 0
            mask[alpha_zero] = 0
        elif name == 'eyeball_H':
            idx = segmentation['eyeball_H'].astype(bool) == True
            mask[idx] = 0
            idx = segmentation['eyelid1_B'].astype(bool) == True
            mask[idx] = 0
            mask[alpha_zero] = 0

        images = np.zeros((seg.shape[0], seg.shape[1], 4), dtype=np.uint8) # RGBA
        images[:,:,:3] = img
        images[label_map, 3] = 255
        images[alpha_zero] = 0
        images[mask_idx] = 0
        if out_file is not None:
            mmcv.mkdir_or_exist(ops.join('/' , *out_file.split('/')))
            
            # size = 128
            # padding_img = np.zeros((512, 512, 4), dtype='uint8')
            # padding_img[size:512-size, size:512-size, :] = images
            # padding_mask = np.zeros((512, 512), dtype='uint8')
            # padding_mask[size:512-size, size:512-size] = mask

            mmcv.imwrite(images, ops.join('/' ,*out_file.split('/'), 'output', name + '.png'))
            mmcv.imwrite(mask, ops.join('/' ,*out_file.split('/'), 'mask', name + '.png'))
    mmcv.imwrite(seg, ops.join('/' ,*out_file.split('/'), 'total_mask.png'))

    mmcv.imwrite(img, ops.join('/' ,*out_file.split('/'), 'ori.png'))
def post_processing(fname, label, seg, erosion=False):
    array = np.uint8(seg == label)
    if erosion:
        kernel = np.ones((3, 3), np.uint8)
        array = cv2.erode(array, kernel, iterations=2)  #// make erosion image
   
    if fname == 'mode':
        f = ImageFilter.ModeFilter(size=7)
        label_map = Image.fromarray(array).filter(f)
    elif fname == 'opening':
        kernel = np.ones((3, 3), np.uint8)
        label_map = cv2.morphologyEx(array, cv2.MORPH_OPEN, kernel)
    elif fname == 'gaussian':
        f = ImageFilter.GaussianBlur(radius=5)
        label_map = Image.fromarray(array).filter(f)
    elif fname == 'sharpening':
        f = ImageFilter.filter(ImageFilter.EDGE_ENHANCE_MORE)
        label_map = Image.fromarray(array).filter(f)

    elif fname == 'None':
        label_map = array

    label_map = np.array(label_map) > 0.5
    
    return label_map

