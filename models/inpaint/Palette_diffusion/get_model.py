import sys
import torch
import numpy as np
from torchvision import transforms

from PIL import Image

from models.inpaint.utils import postprocess, set_device
# sys.path.append("models/image_inpainting")

from models.inpaint.Palette_diffusion.models.parser import init_obj


class Palette_diffusion:
    def __init__(self, model_path, **kwargs):
        self.load_model(model_path)
        
        self.tfs = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])

    def load_model(self, model_path):
        network_opt = {
                "name": ["network", "Network"],# // import Network() class / function(not recommend) from default file (default is [models/network.py]) 
                "args": { #// arguments to initialize network
                    "init_type": "kaiming", #// method can be [normal | xavier| xavier_uniform | kaiming | orthogonal], default is kaiming
                    "module_name": "guided_diffusion", #// sr3 | guided_diffusion
                    "unet": {
                        "in_channel": 6,
                        "out_channel": 3,
                        "inner_channel": 64,
                        "channel_mults": [
                            1,
                            2,
                            4,
                            8
                        ],
                        "attn_res": [
                          #  // 32,
                            16
                           # // 8
                        ],
                        "num_head_channels": 32,
                        "res_blocks": 2,
                        "dropout": 0.2,
                        "image_size": 256
                    },
                    "beta_schedule": {
                        "train": {
                            "schedule": "linear",
                            "n_timestep": 2000,
                            #// "n_timestep": 10, // debug
                            "linear_start": 1e-6,
                            "linear_end": 0.01
                        },
                        "test": {
                            "schedule": "linear",
                            "n_timestep": 1000,
                            "linear_start": 1e-4,
                            "linear_end": 0.09
                        }
                    }
                }
            }
        # networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]
        self.model = init_obj(network_opt, default_file_name='models.network', init_type='Network')
        
        param_key_g = 'params'

        pretrained_model = torch.load(model_path,map_location = lambda storage, loc: set_device(storage))
        self.model.load_state_dict(pretrained_model, strict=True)

        self.model.set_new_noise_schedule(device=torch.device('cuda'), phase='test')
        self.model = self.model.cuda()
        



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
            left_cond = left_img*(1. - left_mask) + left_mask*torch.rand(left_img.size(), dtype=torch.float)

            
            right_img = self.tfs(Image.fromarray(right_img)).type(torch.float32)
            right_mask = torch.from_numpy(right_mask).unsqueeze(0).type(torch.float32)
            right_cond = right_img*(1. - right_mask) + right_mask*torch.rand(right_img.size(), dtype=torch.float)

            image_list.extend([left_img, right_img])
            mask_list.extend([left_mask, right_mask])
            conda_list.extend([left_cond, right_cond])

        image_list = torch.stack(image_list).cuda()
        mask_list = torch.stack(mask_list).cuda()
        conda_list = torch.stack(conda_list).cuda()

        
        with torch.no_grad():
        
            output, _ = self.model.forward(conda_list, y_t=conda_list, 
                                    y_0=image_list, mask=mask_list, sample_num=8)
            output = postprocess(output.detach().float().cpu())

        return output


  

       