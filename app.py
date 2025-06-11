import streamlit as st
import torch
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.model_loader import load_bs_model, load_heston_model

# Set page config
st.set_page_config(
    page_title="PINN Option Pricing",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stSlider > div > div > div {
        background-color: #4CAF50;
    }
    .stRadio > div {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 PINN Option Pricing")
st.markdown("### Physics-Informed Neural Network for Option Pricing")

# Check if model files exist
bs_model_path = "saved_models/black_scholes_model.pth"
heston_model_path = "saved_models/heston_model.pth"
bs_equation_path = "results/bs_equation.png"
heston_equation_path = "results/heston_equation.png"

if not os.path.exists(bs_model_path):
    st.error(f"Black-Scholes model file not found at {bs_model_path}")
    st.stop()
if not os.path.exists(heston_model_path):
    st.error(f"Heston model file not found at {heston_model_path}")
    st.stop()

# Sidebar for model selection
model_type = st.sidebar.radio("Choose Model", ("Black-Scholes", "Heston"))

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Parameters")
    S = st.slider("Stock Price ($)", 1.0, 200.0, 100.0, help="Current stock price")
    t = st.slider("Time to Maturity (years)", 0.01, 1.0, 0.5, help="Time until option expiration")

    if model_type == "Black-Scholes":
        try:
            model = load_bs_model()
            input_tensor = torch.tensor([[S, t]], dtype=torch.float32)
        except Exception as e:
            st.error(f"Error loading Black-Scholes model: {str(e)}")
            st.stop()
    else:
        v = st.slider("Volatility (v)", 0.01, 0.5, 0.04, help="Current volatility")
        try:
            model = load_heston_model()
            input_tensor = torch.tensor([[S, v, t]], dtype=torch.float32)
        except Exception as e:
            st.error(f"Error loading Heston model: {str(e)}")
            st.stop()

with col2:
    st.subheader("Results")
    try:
        with torch.no_grad():
            price = model(input_tensor).item()
        
        st.metric(
            label="Option Price",
            value=f"${price:.4f}",
            delta=None
        )
        
        # Display model equation
        if model_type == "Black-Scholes":
            if os.path.exists(bs_equation_path):
                st.image(bs_equation_path, caption="Black-Scholes PDE")
            else:
                st.warning("Black-Scholes equation image not found")
        else:
            if os.path.exists(heston_equation_path):
                st.image(heston_equation_path, caption="Heston PDE")
            else:
                st.warning("Heston equation image not found")
            
    except Exception as e:
        st.error(f"Error calculating option price: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and PyTorch")
