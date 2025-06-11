import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.model_loader import load_bs_model, load_heston_model

# Set page config
st.set_page_config(
    page_title="PINN Option Pricing",
    layout="wide"
)

st.title("PINN Option Pricing")
st.markdown("Physics-Informed Neural Network for Option Pricing")

# Sidebar for model selection and input
st.sidebar.header("Choose Model")
model_type = st.sidebar.radio("Model", ("Black-Scholes", "Heston"))

st.sidebar.header("Input Parameters")
S = st.sidebar.slider("Stock Price ($)", 1.0, 200.0, 100.0)
t = st.sidebar.slider("Time to Maturity (years)", 0.01, 1.0, 0.5)

if model_type == "Black-Scholes":
    model = load_bs_model()
    input_tensor = torch.tensor([[S, t]], dtype=torch.float32)
else:
    v = st.sidebar.slider("Volatility (v)", 0.01, 0.5, 0.04)
    model = load_heston_model()
    input_tensor = torch.tensor([[S, v, t]], dtype=torch.float32)

with torch.no_grad():
    price = model(input_tensor).item()

# Layout: left for price, right for plot
tab1, tab2 = st.columns([1, 2])

with tab1:
    st.subheader("Results")
    st.metric(label="Option Price", value=f"${price:.4f}")

with tab2:
    st.subheader(f"{model_type} PINN Option Price Surface")
    # Generate dynamic surface plot
    S_range = np.linspace(1, 200, 50)
    t_range = np.linspace(0.01, 1.0, 50)
    S_grid, t_grid = np.meshgrid(S_range, t_range)
    if model_type == "Black-Scholes":
        input_grid = torch.tensor(np.stack([S_grid.ravel(), t_grid.ravel()], axis=1), dtype=torch.float32)
        with torch.no_grad():
            price_grid = model(input_grid).numpy().reshape(S_grid.shape)
    else:
        v_val = v
        v_grid = np.full_like(S_grid, v_val)
        input_grid = torch.tensor(np.stack([S_grid.ravel(), v_grid.ravel(), t_grid.ravel()], axis=1), dtype=torch.float32)
        with torch.no_grad():
            price_grid = model(input_grid).numpy().reshape(S_grid.shape)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S_grid, t_grid, price_grid, cmap='viridis' if model_type=="Black-Scholes" else 'plasma')
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Option Price')
    st.pyplot(fig)

st.markdown("---")
st.markdown("Built with Streamlit and PyTorch")
