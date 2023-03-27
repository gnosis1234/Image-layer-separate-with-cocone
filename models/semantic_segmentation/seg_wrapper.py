import pdb

import torch
import numpy as np
from PIL import Image
from models.image_model import ImageModel
from models.semantic_segmentation.mmseg.apis import inference_segmentor, init_segmentor
from models.tools.processing import rgba2rgb, post_processing
import io 

class Segmentation(ImageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        

    def load_model(self):
        model = None
        if self.model_id in ['segformer_eye', 'deeplabv3plus_eye']:
            model = init_segmentor(self.config, self.model_path)

        self.model = model
    
    def load_image(self, img_path=None, bytes=None):
        if img_path:
            img = Image.open(img_path)
        elif bytes:
            img = Image.open(io.BytesIO(bytes))
        else:
            raise RuntimeError("img_path or bytes must be specified")
        return rgba2rgb(np.array(img), background=(0,0,0))

    def seperate_layers(self, image, **kwargs):
        images = [rgba2rgb(image, background=(0,0,0)) ]
      
        self.model.eval()
        with torch.no_grad():
            masks = inference_segmentor(self.model, imgs=images)
              
        seg_images, target_masks = self.mask_generator(images, masks)
        
        return seg_images, target_masks

    
    def mask_generator(self, imgs, segs):

        class_name = ['background', 'eyeballH', 'eyelidB', 'eyelidF']
        target_images = dict((name, []) for i, name in enumerate(class_name))
        target_labels = dict((name, []) for i, name in enumerate(class_name))
       
        
        for img, seg in zip(imgs, segs):
            alpha_zero = np.average(img, axis=2) < 20
            segmentation = {'image': {}, 'mask': {}}
            pp_img_method = {'eyelidB': ['erode_2_2', 'mode'], 'eyeballH': ['erode_2_2', 'mode'], 'eyelidF':['mode']}
            pp_mask_method = {'eyelidB': ['erode_5_1'], 'eyeballH': ['erode_5_1'], 'eyelidF':['mode']}
            for label, name in enumerate(class_name):
                if label == 0: continue
                segmentation['image'][name] = (np.asarray(post_processing(image=(np.uint8(label==seg)), methods=pp_img_method[name])) > 0.5).astype(bool)
                segmentation['mask'][name] = (np.asarray(post_processing(image=(np.uint8(label==seg)), methods=pp_mask_method[name])) > 0.5).astype(bool)
         
            for label, name in enumerate(class_name):
                if label == 0: continue
        
                mask = np.ones((seg.shape[0], seg.shape[1]))
                if name == 'eyelidB':
                    mask[segmentation['mask']['eyelidB']] = 0
                    mask[alpha_zero] = 0
                elif name == 'eyeballH':
                    mask[segmentation['mask']['eyeballH']] = 0
                    mask[segmentation['mask']['eyelidB']] = 0
                    mask[alpha_zero] = 0

                images = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # RGBA
                images[segmentation['image'][name]] = img[segmentation['image'][name]]
                
                target_images[name] += [images]
                target_labels[name] += [mask]


        return target_images, target_labels
