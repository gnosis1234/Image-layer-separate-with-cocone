import pdb

import torch

import numpy as np
from PIL import Image
import cv2

import io
from models.image_model import ImageModel
from models.inpaint.Palette_diffusion.get_model import Palette_diffusion
from models.inpaint.mmseries.get_model import MMedit_inpaint

class Inpainting(ImageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        

    def load_model(self):
        self.model = None        
        self.model_id, self.part_name = self.model_id.split('_') # palette_eyeball, palette_eyelid
        if self.model_id == 'palette': 
            self.model = Palette_diffusion(self.model_path)
            self.input_len = 10
        elif self.model_id == 'aot-gan':
            self.input_len = 10
            self.model = MMedit_inpaint(self.model_path, self.config)
   
    def load_image(self, img_path=None, bytes=None):
        if img_path:
            img = Image.open(img_path)
        elif bytes:
            img = Image.open(io.BytesIO(bytes))
        else:
            raise RuntimeError("img_path or bytes must be specified")
        return np.array(img.convert("RGB")) 

    def seperate_layers(self, images=None, masks=None, **kwargs):
    
        image = images['image']
        mask = (cv2.cvtColor(images['mask'], cv2.COLOR_RGBA2GRAY) // 255) | (cv2.cvtColor(masks, cv2.COLOR_RGB2GRAY) // 255)
        
        images = [image]
        masks = [mask]
        results = []
        if self.part_name == 'eyelidF':
            
            for image in images:
                h,w,c = np.shape(image)
                left_img, right_img = image[h//4:h-h//4, :w//2, :],image[h//4:h-h//4, w//2:, :]
                results.extend([Image.fromarray(img) for img in [left_img, right_img]])
            return results
      

        with torch.no_grad():
            self.input_len = self.input_len if len(images) > self.input_len else len(images)
            for interval in range(0, len(images), self.input_len):
                input_range = interval + self.input_len if interval + self.input_len < len(images) else len(images)
                inputs_img = images[interval:input_range]
                inputs_mask = masks[interval:input_range]
            
                outputs = self.model.forward(inputs_img, inputs_mask)
                results.extend(outputs)
                

        return results
