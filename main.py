import pdb

import gradio as gr
import torch
from zipfile import ZipFile
import numpy as np
from models import get_model

device = torch.device("cpu")


def query_image(img, model_type, model_path, model_id, config):
    # model = get_model(model_type=model_type, model_path='./flagged/checkpoint/segformer/segformer_iter_20000.pth', model_id='segformer_eye', config='./flagged/checkpoint/segformer/segformer.py')
    
    model = get_model(model_type=model_type, model_path=model_path.name, model_id=model_id, config=config.name)
    # Image size 512 x 512 x 3
    with torch.no_grad():
        outputs = model.seperate_layers(image=img)

    # results = visualize_instance_seg_mask(outputs.cpu().detach().numpy())
    return outputs[0]['eyeballH'][0], outputs[0]['eyelidB'][0], outputs[0]['eyelidF'][0], outputs[1]['eyeballH'][0], outputs[1]['eyelidB'][0]

def query_inpaint(img, mask, model_type, model_path, model_id, config):
    
    model = get_model(model_type=model_type, model_path=model_path.name, model_id='aot-gan_eyeballH', config=config.name)
    with torch.no_grad():
        outputs = model.seperate_layers(images=img, masks=mask)

    return outputs

with gr.Blocks() as demo:
    gr.Markdown("Cocone M Algorhtim")
    with gr.Tab("Segmentation"):
        labels = ['eyeballH', 'eyelidB', 'eyelidF', 'eyeballH_mask', 'eyelidB_mask']
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="image")
                model_type = gr.Dropdown(choices=["segmentation"], label="model type", value="segmentation")
                model_path = gr.File(file_types=[".pth", ".pt"], label="model path", value='./flagged/checkpoint/segformer/segformer_iter_20000.pth')
                model_id = gr.Dropdown(choices=["segformer_eye", "deeplabv3plus_eye"], label="model id", value='segformer_eye')
                config = gr.File(file_types=[".py"], label="config", value='./flagged/checkpoint/segformer/segformer.py')

                segmentation_button = gr.Button("Segmentation")
            with gr.Column():
                outputs = [gr.Image(label=labels[i]) for i in range(5)]
        gr.Examples([['dataset/segmentation/XRocriJhvvBrZu2SXD73m2_002.png', 'segmentation', './flagged/checkpoint/segformer/segformer_iter_20000.pth', 'segformer_eye', './flagged/checkpoint/segformer/segformer.py'],                
                     ], inputs=[image, model_type, model_path, model_id, config])
    with gr.Tab("Inpaint"):
        with gr.Row():
            with gr.Column():
                image2 = gr.Image(tool="sketch", label="image")
                mask2 = gr.Image(label="mask")
                model_type2 = gr.Dropdown(choices=["inpaint"], label="model type", value="inpaint")
                model_path2 = gr.File(file_types=[".pth", ".pt"], label="model path", value='./flagged/checkpoint/aot-gan/aot-gan_eyeball.pth')
                model_id2 = gr.Dropdown(choices=["aot-gan_eyeballH", "aot-gan_eyelidB"], label="model id", value='aot-gan_eyeballH')
                config2 = gr.File(file_types=[".py"], label="config", value='./flagged/checkpoint/aot-gan/AOT-GAN_256x256_4x12_eyeball.py')
                
                image_button = gr.Button("Inpaint")
        
            with gr.Column():
                left_eye = gr.Image(label="left eye")
                right_eye = gr.Image(label="right eye")
        
        gr.Examples([
                     ['dataset/inpaint/eyelidB_image.png', 'dataset/inpaint/eyelidB_mask.png', 'inpaint', './flagged/checkpoint/aot-gan/aot-gan_eyelidB.pth', 'aot-gan_eyelidB', './flagged/checkpoint/aot-gan/AOT-GAN_256x256_4x12_eyelidB.py'],
                     ['dataset/inpaint/eyeballH_image.png', 'dataset/inpaint/eyeballH_mask.png', 'inpaint', './flagged/checkpoint/aot-gan/aot-gan_eyeball.pth', 'aot-gan_eyeballH', './flagged/checkpoint/aot-gan/AOT-GAN_256x256_4x12_eyeball.py']
                     ], inputs=[image2, mask2, model_type2, model_path2, model_id2, config2])
    segmentation_button.click(fn=query_image, 
            inputs=[image, model_type, model_path, model_id, config  ], 
            outputs=outputs,
            # title="Image Segmentation Demo",
            # description = "Please upload an image to see segmentation capabilities of this model",)
        )

    image_button.click(fn=query_inpaint, 
            inputs=[image2, mask2, model_type2, model_path2, model_id2, config2  ], 
            outputs=[left_eye, right_eye],
            )

demo.launch()