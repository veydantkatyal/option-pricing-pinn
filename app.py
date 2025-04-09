import streamlit as st
import torch
from utils.model_loader import load_bs_model, load_heston_model

st.title("PINN Option Pricing")

model_type = st.radio("Choose Model", ("Black-Scholes", "Heston"))
S = st.slider("Stock Price", 1.0, 200.0, 100.0)
t = st.slider("Time to Maturity", 0.01, 1.0, 0.5)

if model_type == "Black-Scholes":
    model = load_bs_model()
    input_tensor = torch.tensor([[S, t]], dtype=torch.float32)
else:
    v = st.slider("Volatility (v)", 0.01, 0.5, 0.04)
    model = load_heston_model()
    input_tensor = torch.tensor([[S, v, t]], dtype=torch.float32)

with torch.no_grad():
    price = model(input_tensor).item()

st.success(f"Option Price = ${price:.4f}")
