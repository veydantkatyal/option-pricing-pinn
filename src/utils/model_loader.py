import torch
from src.models.bs_pinn import PINN
from src.models.heston_pinn import HestonPINN

def load_bs_model():
    model = PINN()
    model.load_state_dict(torch.load("saved_models/black_scholes_model.pth", map_location='cpu'))
    model.eval()
    return model

def load_heston_model():
    model = HestonPINN()
    model.load_state_dict(torch.load("saved_models/heston_model.pth", map_location='cpu'))
    model.eval()
    return model
