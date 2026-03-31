import torch
import torch.nn as nn
from torchvision import models

def build_model(weights_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model.to(device)
