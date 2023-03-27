import pdb 

import torch
from torchvision import transforms

from PIL import Image
import numpy as np



from mmedit.apis import init_model, inpainting_inference
# from mmedit.core import tensor2img
from models.inpaint.utils import postprocess, set_device


class MMedit_inpaint:
    def __init__(self, model_path, config, **kwargs):
        self.config = config
        self.load_model(model_path)
        
        self.tfs = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])

    def load_model(self, model_path):
        self.model = init_model(self.config, model_path, device='cpu')
        
        



    def forward(self, images, masks, **kwargs):
        self.model.eval()
    
        image_list, mask_list, conda_list = [], [], []
         # inference
        for batch_idx, (image, mask) in enumerate(zip(images, masks)):
            # image = images[batch_idx]
            # mask = masks[batch_idx]
            
            h, w, c = np.shape(image)
            # Left
            left_img, left_mask = image[h//4:h-h//4, :w//2, :], mask[h//4:h-h//4, :w//2]
            # Right
            right_img, right_mask = image[h//4:h-h//4, w//2:, :], mask[h//4:h-h//4, w//2:]
            
            left_img = self.tfs(Image.fromarray(left_img)).type(torch.float32)
            left_mask = torch.from_numpy(left_mask).unsqueeze(0).type(torch.float32)
        

            right_img = self.tfs(Image.fromarray(right_img)).type(torch.float32)
            right_mask = torch.from_numpy(right_mask).unsqueeze(0).type(torch.float32)
        

            image_list.extend([left_img, right_img])
            mask_list.extend([left_mask, right_mask])
            

        image_list = torch.stack(image_list)
        mask_list = torch.stack(mask_list)
  
        
        with torch.no_grad():
            # pdb.set_trace()
            output = inpainting_inference(self.model, image_list, mask_list)
            # output = tensor2img(output, min_max=(-1, 1))
            
         
            # output = Image.fromarray(output)
            output = output.detach().float().cpu()
            output = postprocess(output)

        return output


  

       