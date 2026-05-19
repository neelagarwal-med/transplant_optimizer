# Kidney Utilization Prediction System

This repository contains the source code for a clinical decision support tool designed to assist transplant teams in the assessment of marginal deceased donor kidneys. The application provides a "second opinion" by utilizing machine learning to predict organ utilization probability based on historical OPTN (Organ Procurement and Transplantation Network) data.

## Key Features
* **Calibrated Risk Engine:** A LightGBM classification pipeline calibrated via Isotonic Regression to provide statistically accurate probability outputs.
* **Clinical KDPI Calculator:** Built-in engine implementing the Rao et al. (2009) formula for rapid KDRI/KDPI estimation.
* **Explainable AI (XAI):** Integration with SHAP (SHapley Additive exPlanations) to provide clinicians with the specific physiological drivers influencing each prediction.
* **Data-Driven Workflow:** Optimized for high-stakes decision-making environments, such as during donor offer evaluations.

## Tech Stack
* **Frontend:** Streamlit
* **Modeling:** LightGBM, Scikit-Learn
* **Explainability:** SHAP
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly

## Installation
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
