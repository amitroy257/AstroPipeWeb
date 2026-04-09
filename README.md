# 🔭 AstroPipe: Scientific ML Pipeline

AstroPipe is a modular, config-driven machine learning pipeline for astronomical image classification (Galaxies vs. Stars). It serves as a clean software architecture prototype for research-grade AI workflows.

## 🚀 Quick Start (VS Code)

Follow these steps to get the dashboard running on your local machine:

1. **Open Folder:** Open the `AstroPipe-main` folder in VS Code.

2. **Install Dependencies:** Open the terminal (`Ctrl + ~`) and run:
   ```bash
   pip install -r requirements.txt

3. **Launch the Dashboard:** To open the web interface for training and prediction, run:
   ```bash
   python -m streamlit run app.py

4. **Run via CLI:** To run the training pipeline directly in the terminal, use:
   ```bash
   python cli.py --config config.yaml