# Car Insurance Claim Predictor

![Python Version](https://img.shields.io/badge/python-3.14.5-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136.3-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-blue)

A complete, production-grade Machine Learning pipeline that predicts the likelihood of a car insurance claim and estimates the potential claim cost for policyholders.

This project refactors an exploratory Jupyter notebook (`main.ipynb`) into a modular, maintainable, and deployable application following industry best practices.

## 📖 Project Overview

Car insurance providers rely heavily on risk assessment to price premiums accurately and maintain profitability. This project solves a dual-objective problem for an auto insurance company:
1. **Risk Classification**: Accurately predict whether a given policyholder will file a claim (Binary Classification).
2. **Cost Estimation**: Given that a claim will be filed, estimate the expected monetary value of that claim (Regression).

By solving these two objectives, the business can automatically flag high-risk customers, adjust their premiums, and maintain adequate financial reserves for expected payouts.

### 📊 Dataset Context
The data consists of historical records of policyholders. Features include:
- **Demographics**: Age, Gender, Education, Marital Status, Occupation
- **Financial Status**: Income, Home Value, Years on Job
- **Driving History**: License points, Number of past claims, Value of past claims, Commute distance
- **Vehicle Information**: Vehicle Type, Vehicle Age, Vehicle Value, Type of Use (Commercial/Private)

The target variables are `claim_flag` (1 if a claim was made, 0 otherwise) and `clm_amt` (the monetary amount of the claim).

## 🚀 Features

- **End-to-End ML Pipeline**: Automated data cleaning, imputation, and feature engineering (One-Hot Encoding, Ordinal Encoding, Scaling) using scikit-learn `Pipeline` and `ColumnTransformer`.
- **Custom Transformers**: Object-oriented implementations of data transformations (`ColumnDropper`) to seamlessly integrate into the scikit-learn ecosystem and prevent multicollinearity.
- **Dual-Model Architecture**:
  - **Classification**: XGBoost Classifier to predict whether a driver will make a claim (claim probability).
  - **Regression**: SGD Regressor to estimate the expected claim amount for high-risk profiles.
- **REST API**: A robust FastAPI backend exposing the prediction models with Pydantic schema validation.
- **Interactive Dashboard**: A modern, premium Streamlit web application for real-time predictions and profile risk analysis.
- **Dockerized**: Easy containerization with optimized layer caching for quick deployments.

## 📂 Project Structure

```text
Car-Insurance-Claim/
├── data/
│   └── car_insurance_claim.csv# Original raw dataset
├── models/                    # Serialized pipeline & model artifacts (.joblib)
├── src/                       # Modular source code
│   ├── __init__.py
│   ├── config.py              # Configuration & feature lists
│   ├── custom_transformers.py # Custom scikit-learn transformers
│   ├── preprocessing.py       # Imputation & encoding pipelines
│   ├── train.py               # Training script
│   └── predict.py             # Inference wrapper
├── app/                       # FastAPI deployment service
│   ├── main.py                # REST API endpoints
│   └── schemas.py             # Pydantic data models
├── dashboard/                 # Streamlit Showcase Web UI
│   └── app.py                 # Streamlit application
├── tests/                     # Unit Tests (pytest)
│   ├── test_transformers.py
│   └── test_api.py
├── Dockerfile                 # Docker container config
├── requirements.txt           # Production dependencies
├── dev-requirements.txt       # Development/testing dependencies
└── main.ipynb                 # Original research notebook
```

## 🛠️ Quick Start

### 1. Installation

Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Training the Models

Before running the application, you must train the models to generate the serialized `.joblib` artifacts. The data must be located at `data/car_insurance_claim.csv`.

```bash
python -m src.train
```

This will output the evaluation metrics (Classification Report, MSE, MAE) and save the trained preprocessor and models into the `models/` directory.

### 3. Running the REST API (FastAPI)

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can explore the interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

### 4. Running the Web Dashboard (Streamlit)

In a new terminal, start the Streamlit application:

```bash
streamlit run dashboard/app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`. 

## 🐳 Docker Deployment

You can run the entire service via Docker. The provided Dockerfile is configured to run the FastAPI backend. 

```bash
# Build the image
docker build -t car-insurance-api .

# Run the container
docker run -p 8000:8000 car-insurance-api
```

## 🧪 Testing

To run the unit tests, install the development dependencies and run `pytest`:

```bash
pip install -r dev-requirements.txt
pytest tests/
```

## 📝 Methodology Highlights

- **Data Leakage Prevention**: In the original notebook, the preprocessor was fitted on the entire dataset (including the test set), leading to data leakage. The refactored `src/train.py` strictly uses `.fit_transform()` on the training set and `.transform()` on the test set.
- **Multicollinearity Handling**: VIF analysis revealed perfect multicollinearity among the One-Hot Encoded features. A custom `ColumnDropper` was integrated directly into the `Pipeline` to dynamically drop the reference baseline categories (e.g., `occupation_Blue Collar`).
