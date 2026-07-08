# 👁️👄 DeficiVision: Intelligent Nutritional Health Assessment

An AI-powered diagnostic prototype utilizing a hybrid ResNet50 (CNN) and Vision Transformer (ViT) pipeline to detect vitamin deficiencies from clinical images of eyes, lips, tongues, gums, skin, and hair. 

## Features
* **Hybrid Architecture:** Combines the spatial feature extraction of CNNs with the global context understanding of Vision Transformers.
* **Soft-Voting Ensemble:** Averages probability maps to output high-confidence predictions.
* **Interactive UI:** Built with Streamlit for a clean, clinical-style dashboard including dynamic health recommendations.

## Installation & Setup

1. Clone the repository
```bash
git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
cd YOUR-REPO-NAME

2. Install dependencies
pip install -r requirements.txt

3. Download the Model Weights
Because the trained .pth files are too large for standard GitHub hosting, they are hosted securely on Google Drive.

Download cnn_vitamin_model.pth here: https://drive.google.com/file/d/1GXDhdwP2Xmwx_nUt5aYi8Mp4GJ4a_Dqt/view
Download vit_vitamin_model.pth here: https://drive.google.com/file/d/1vHg0QHWgEJJujWsEpD50mdKe5MxpxCHr/view

4. Run the Application
streamlit run app.py
