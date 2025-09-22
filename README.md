# MammoMind: Interpretable Breast Cancer Detection

MammoMind is my Final Year Project for CM3070.  
It is an **attention-driven deep learning system** for breast cancer detection on mammograms, designed with **uncertainty estimation** and **explainability (Grad-CAM)** to support radiologists in decision-making.

---

##  Features
- **Patch-based CNN** for mammography
- **Attention (SE/CBAM) blocks** to improve sensitivity
- **Monte Carlo Dropout** for uncertainty estimation
- **Grad-CAM overlays** for visual interpretability
- **Streamlit web app** with:
  - Image upload & prediction  
  - Confidence & uncertainty display  
  - Grad-CAM heatmaps & outlines  
  - Interactive metrics dashboard  
  - Informational "About" page  

---

## Running the App - - 
please read requirements.txt . Intall all necessary Python libraries first.

From the project root:

streamlit run app/app.py

Then, The app will launch in your browser.


## Disclaimer

MammoMind is a research and educational prototype.
It is not for clinical use or diagnostic decision-making.
