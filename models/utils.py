from models.semantic_segmentation import Segmentation
from models.inpaint import Inpainting
def get_model(model_type, model_path, model_id, config=None, **kwargs):
    if model_type == 'segmentation':
        return Segmentation(model_path, model_id, config, **kwargs)
    elif model_type == 'inpaint':
        return Inpainting(model_path, model_id, config, **kwargs)
        