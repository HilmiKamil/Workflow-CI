import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===============================
# LOAD DATA HASIL PREPROCESSING
# ===============================
X_train = np.load("heart_disease_preprocessing/X_train.npy")
y_train = np.load("heart_disease_preprocessing/y_train.npy")
X_test = np.load("heart_disease_preprocessing/X_test.npy")
y_test = np.load("heart_disease_preprocessing/y_test.npy")

# ===============================
# SETTING MLFLOW
# ===============================
mlflow.set_experiment("Heart Disease Classification")
mlflow.sklearn.autolog()

with mlflow.start_run():

    # ===========================
    # TRAIN MODEL
    # ===========================
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # ===========================
    # PREDIKSI
    # ===========================
    y_pred = model.predict(X_test)

    # ===========================
    # HITUNG METRIK
    # ===========================
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # ===========================
    # LOG PARAMETER
    # ===========================
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)

    # ===========================
    # LOG METRIK
    # ===========================
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # ===========================
    # LOG & REGISTER MODEL (KUNCI KRITERIA 3)
    # ===========================
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="HeartDiseaseModel"
    )

    print("Training selesai, model ter-register ke MLflow")