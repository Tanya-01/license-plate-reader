from django.apps import AppConfig
from django.conf import settings
import os
import pickle
from .yolo.models import Darknet
import torch

class ModelConfig(AppConfig):
    name='model'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(settings.MODELS_ROOT,'model.pth')
    config_path = os.path.join(settings.MODELS_ROOT,'config.cfg')
    model = Darknet(config_path=config_path).to(device)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()