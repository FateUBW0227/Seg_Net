import torch
from models.UnetModel4 import UNet3D

def LoadModel(model_config, modelPath=None):
    model = UNet3D(**model_config)
    if not modelPath is None:
        ckpt = torch.load(modelPath)
        model.load_state_dict(ckpt['state_dict'])
    return model
