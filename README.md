# PINN Option Pricing Web App

A web application for option pricing using Physics-Informed Neural Networks (PINNs) implemented in PyTorch and Streamlit.

## Features

- Black-Scholes model pricing
- Heston model pricing
- Interactive UI with real-time calculations
- Visual representation of pricing equations

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── Procfile           # Deployment configuration
├── src/
│   ├── models/        # Neural network model definitions
│   └── utils/         # Utility functions
├── saved_models/      # Trained model weights
└── results/           # Equation images and results
```

## Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## Deployment

This app can be deployed on various platforms:

### Streamlit Cloud
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

### Heroku
1. Create a Heroku account
2. Install Heroku CLI
3. Run:
```bash
heroku create your-app-name
git push heroku main
```

## Requirements

- Python 3.8+
- PyTorch
- Streamlit
- NumPy
- Matplotlib
- Pillow

## License

MIT License 