
# 🏨 Hotel Reservation Intelligence System

A professional **End-to-End Machine Learning** project designed to predict hotel reservation cancellations. The system integrates a high-performance **XGBoost** model with a full **MLOps** pipeline, served via **FastAPI**, and visualized through an interactive **Streamlit** dashboard — all fully **Dockerized**.

---

## 🚀 Key Features

* **Predictive Analytics:** Accurately predicts booking cancellations using an optimized **XGBoost** model.
* **MLOps Pipeline:** Managed training, evaluation, and deployment workflows using **DVC pipelines**.
* **Real-time Inference:** Low-latency FastAPI backend with automated feature engineering.
* **Interactive Dashboard:** Streamlit UI with probability scores, risk alerts, and visual insights.
* **Environment Agnostic:** Fully containerized using **Docker** for easy deployment.

---

## 🛠️ Tech Stack

* **ML & Data:** Python, Pandas, Scikit-learn, XGBoost
* **MLOps:** MLflow (Experiment Tracking), DVC (Pipeline Workflow Management)
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Frontend:** Streamlit, Requests
* **DevOps:** Docker

---

## 🔄 MLOps Workflow

### 📈 Experiment Tracking with MLflow

* Logs every training cycle, including hyperparameters (`n_estimators`, `max_depth`) and metrics (Accuracy, F1-Score).
* Ensures the best performing model is promoted to production.
* **View logs:**

```bash
mlflow ui
```

### 📦 DVC Pipelines

* Organizes **training, evaluation, and deployment** steps into reproducible pipelines.
* Ensures that running a pipeline produces consistent outputs without manually running scripts in order.
* Example:

```bash
dvc repro
```

---

## 📋 Getting Started (Local Setup)

### 1️⃣ Clone & Install Dependencies

```bash
git clone https://github.com/your-username/hotel-reservation-ml.git
cd hotel-reservation-ml
pip install -r requirements.txt
```

### 2️⃣ Run the DVC Pipeline (Optional)

```bash
dvc repro
```

### 3️⃣ Run with Docker (Recommended)

```bash
# Build the Docker image
docker build -t hotel-reservation-app .

# Run the container
docker run -p 8000:8000 hotel-reservation-app
```

### 4️⃣ Launch Streamlit Dashboard

```bash
streamlit run frontend_app.py
```

---

## 📁 Project Structure

```
.
├── Dockerfile                  # Containerization script
├── dvc.yaml                    # DVC pipeline configuration
├── dvc.lock                    # DVC lock file (State tracking)
├── config/
│   └── config.yaml             # Project-wide configuration parameters
├── data/
│   ├── raw/                    # Original, immutable dataset
│   │   └── Hotel Reservations.csv
│   └── processed/              # Cleaned and feature-engineered data
├── models/                     # Trained model artifacts (.pkl)
│   ├── prepipeline.pkl         # Preprocessing pipeline object
│   └── xgb_model.pkl           # Final XGBoost model
├── artifact/                   # Visualizations & evaluation metrics
│   ├── conf_matrix.png         # Confusion Matrix plot
│   └── roc_auc.png             # ROC-AUC curve plot
├── logs/                       # Application & Training logs
├── mlruns/                     # MLflow experiment tracking files
├── mlflow.db                   # MLflow SQL database (Local tracking)
├── src/                        # Core Source Code
│   ├── api/                    # FastAPI backend implementation
│   │   └── app.py
│   ├── schemas/                # Pydantic data validation schemas
│   │   └── input_schema.py
│   ├── load_data.py            # Data ingestion script
│   ├── validate_data.py        # Data validation logic (Pandera)
│   ├── feature_engineering.py   # Transformation & feature creation
│   ├── preprocessing.py        # Scaling and encoding logic
│   ├── xgboost_model.py        # Model training and hyperparameter tuning
│   ├── evaluation.py           # Metrics calculation & plot generation
│   └── utils.py                # Common helper functions
├── frontend_app.py             # Streamlit interactive dashboard
├── notebooks/                  # Experimental analysis & EDA
│   ├── EDA.ipynb               # Exploratory Data Analysis
│   └── ML.ipynb                # Model prototyping
├── requirements.txt            # Python dependencies
└── tests/                      # Unit tests for core functions
    └── test_main.py
```
---

## 🧠 How It Works

1. **Frontend:** Users enter booking details (Lead time, price, etc.) via Streamlit.
2. **API Call:** Data is sent as JSON to FastAPI `/predict` endpoint.
3. **Inference:** API computes features (e.g., `total_nights`) and calls the trained **XGBoost** model.
4. **Feedback:** Prediction and probability score are returned and displayed in Streamlit with clear visual indicators.

---

## 📈 Model Performance (Example)

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.85  |
| F1-Score  | 0.89  |
| Precision | 0.85  |
| Recall    | 0.94  |


---

## 🛠️ Future Improvements

* Integration with **CI/CD** pipelines for automated retraining and deployment.
* Advanced **feature engineering** for more accurate predictions.
* Real-time **alerts & notifications** for high-risk cancellations.

---
