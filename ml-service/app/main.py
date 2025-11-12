from fastapi import FastAPI, UploadFile, Form, File, HTTPException, BackgroundTasks, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from typing import List, Optional
from sqlalchemy import create_engine, text
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import random
import json

from app.preprocess import clean_text, preprocess_liar_dataset
from app.db import (
    save_news_and_labels, 
    save_explanation,
    create_user, verify_user, get_user_by_email, save_feedback,
    get_unapplied_feedback_params, mark_feedback_applied
)
from app.models import FakeNewsClassifier, BertTinyClassifier
from app.utils.xai import ExplainableModel

app = FastAPI(title="FakeNews ML Service")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Моделі ======
models = {
    "logreg": FakeNewsClassifier(),
    "bert-tiny": BertTinyClassifier()
}

def get_explainer(model_name: str):
    if model_name == "logreg":
        return ExplainableModel("logreg")
    elif model_name == "bert-tiny":
        return ExplainableModel("bert-tiny")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'")

model_status = {name: {"running": False, "metrics": None, "ready": False} for name in models.keys()}

def get_model(name: str):
    if name not in models:
        raise HTTPException(status_code=400, detail=f"Unknown model '{name}'")
    return models[name]

# ====== Pydantic Models ======
class PredictRequest(BaseModel):
    news_text: str
    model_name: str = Field("logreg", description="Назва моделі: logreg або bert-tiny")

class AnalyzeRequest(BaseModel):
    model_name: str = Field("logreg", description="Назва моделі: logreg або bert-tiny")
    test_size: float = Field(0.3, alias="testSize")
    max_iter: int = Field(1000)
    C: float = Field(5.0)
    solver: str = Field("liblinear")

    class Config:
        allow_population_by_field_name = True

class FeedbackRequest(BaseModel):
    news_id: Optional[int] = None  # необов'язково, якщо зміна параметра
    type: str
    payload: dict

# ====== Health ======
@app.get("/health")
def health():
    return {"status": "ok"}

class RegisterRequest(BaseModel):
    role: str = Field(..., description="user або researcher")
    name: str
    email: str
    password: str
    org: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str

# ====== Preprocess ======
@app.post("/preprocess")
async def preprocess(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            # Зберігаємо тимчасовий TSV
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile(delete=False, suffix=".tsv") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            # Обробляємо як TSV
            news_df, label_df = preprocess_liar_dataset(tmp_path, is_train=True)
            save_news_and_labels(news_df, label_df)

            results.append({
                "filename": file.filename,
                "rows": len(news_df)
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return {"status": "preprocessed", "files": results}


# ====== Training ======
@app.post("/analyze")
def analyze_all(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    ml_model = get_model(request.model_name)
    status = model_status[request.model_name]

    if status["running"]:
        return {"status": "already_running", "model": request.model_name}

    def background_train():
        model_status[request.model_name]["running"] = True
        try:
            unapplied_params_feedback = get_unapplied_feedback_params(model_name=request.model_name)
            current_params = request.dict()
            feedback_ids_to_mark = []

            if unapplied_params_feedback:
                print(f"Applying {len(unapplied_params_feedback)} unapplied parameter feedbacks for model {request.model_name}")
                for feedback_record in unapplied_params_feedback:
                    feedback_params = feedback_record["payload"].get("params", {})
                    for param_key, param_value in feedback_params.items():
                        if param_key in current_params:
                            try:
                                if isinstance(current_params[param_key], int):
                                    current_params[param_key] = int(param_value)
                                elif isinstance(current_params[param_key], float):
                                    current_params[param_key] = float(param_value)
                                else:
                                    current_params[param_key] = param_value
                            except ValueError:
                                print(f"Warning: Could not convert feedback param '{param_key}' value '{param_value}' to expected type. Skipping.")
                                continue
                        else:
                            print(f"Warning: Feedback param '{param_key}' not found in model's analyze request. Skipping.")
                            continue
                    feedback_ids_to_mark.append(feedback_record["id"])

            if hasattr(ml_model, "run_training"):
                ml_model.run_training(
                    test_size=current_params.get("test_size", 0.3),
                    max_iter=current_params.get("max_iter", 1000),
                    C=current_params.get("C", 5.0),
                    solver=current_params.get("solver", "liblinear")
                )
                metrics = ml_model.train_metrics
            elif hasattr(ml_model, "train"):
                metrics = ml_model.train(test_size=current_params.get("test_size", 0.3))
            else:
                raise HTTPException(status_code=400, detail=f"Модель '{request.model_name}' не підтримує тренування")

            model_status[request.model_name]["metrics"] = metrics
            model_status[request.model_name]["ready"] = True

            if feedback_ids_to_mark:
                mark_feedback_applied(feedback_ids_to_mark)
                print(f"Marked {len(feedback_ids_to_mark)} feedback records as applied.")

        finally:
            model_status[request.model_name]["running"] = False

    background_tasks.add_task(background_train)
    return {"status": "training_started", "model": request.model_name}

@app.get("/analyze/status")
def analyze_status(model_name: str = Query("logreg")):
    status = model_status.get(model_name)
    if not status:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'")
    return status

# ====== Prediction ======
@app.post("/predict")
def predict(req: PredictRequest):
    ml_model = get_model(req.model_name)
    if not ml_model.is_trained():
        raise HTTPException(status_code=400, detail=f"Модель '{req.model_name}' ще не натренована.")
    clean = clean_text(req.news_text)
    label, prob = ml_model.predict(clean)
    return {"model": req.model_name, "label": label, "probability": prob}

# ====== Random Predict + XAI ======
@app.get("/random_predict")
def random_predict(model_name: str = Query("logreg")):
    ml_model = get_model(model_name)
    if not ml_model.is_trained():
        raise HTTPException(status_code=400, detail=f"Модель '{model_name}' ще не натренована.")

    engine = create_engine("postgresql+psycopg2://postgres:100317@db/fakenewsdb")
    with engine.connect() as conn:
        news_ids = conn.execute(text("SELECT id FROM NewsItem")).fetchall()
        if not news_ids:
            return {"error": "База новин порожня"}
        random_id = random.choice(news_ids)[0]
        row = conn.execute(text("""
            SELECT n.text, l.predicted_label, l.confidence, l.label
            FROM NewsItem n
            JOIN Label l ON n.id = l.news_id
            WHERE n.id = :news_id
        """), {"news_id": random_id}).fetchone()
        if not row:
            return {"error": f"Не знайдено прогнозу для новини {random_id}"}

        text_item = row.text
        predicted_label = "real" if row.predicted_label == 1 else "fake"
        true_label = "real" if row.label == 1 else "fake"

        explanations_created = []
        explainer = get_explainer(model_name)
        for method in ["SHAP", "IG", "LIME"]:
            try:
                if method == "SHAP":
                    shap_result = explainer.explain_shap(text_item)
                    values = shap_result[0]["scores"] if isinstance(shap_result, list) else shap_result["scores"]
                elif method == "IG" and model_name=="bert-tiny":
                    ig_result = explainer.explain_ig(text_item)
                    values = ig_result["scores"]
                elif method == "LIME" and model_name=="bert-tiny":
                    lime_result = explainer.explain_lime(text_item)
                    values = lime_result["scores"]
                else:
                    continue
                fidelity = 1.0  # можна додати обчислення fidelity, як у вашому прикладі
                save_explanation(random_id, method, values, fidelity)
                explanations_created.append(method)
            except Exception as e:
                print(f"⚠️ Explanation for {method} failed: {e}")

        return {
            "news_id": random_id,
            "model": model_name,
            "text": text_item[:300]+"..." if len(text_item)>300 else text_item,
            "prediction": {"predicted_label": predicted_label, "probability": float(row.confidence)},
            "true_label": true_label,
            "created_explanations": explanations_created
        }

# ====== Interpretability ======
@app.post("/interpret/{method}")
def interpret(method: str, news_id: int, model_name: str = Query("logreg")):
    engine = create_engine("postgresql+psycopg2://postgres:100317@db/fakenewsdb")
    query = text("""
        SELECT e.payload, e.fidelity, l.predicted_label
        FROM Explanation e
        JOIN Embedding em ON e.news_id = em.news_id
        JOIN Label l ON e.news_id = l.news_id
        WHERE e.news_id = :news_id
        AND em.model_id = :model_id
        AND e.method = :method
        ORDER BY e.id DESC
        LIMIT 1;
    """)
    df = pd.read_sql(query, engine, params={
        "news_id": news_id,
        "model_id": model_name,
        "method": method.upper()
    })
    if df.empty:
        raise HTTPException(status_code=404, detail="Пояснення не знайдено")
    
    value = df.iloc[0]["predicted_label"]
    try:
        value_int = int(value)
    except (ValueError, TypeError):
        value_int = 0
    predicted_label = "real" if value_int == 1 else "fake"
    
    return {
        "news_id": news_id,
        "model": model_name,
        "method": method.upper(),
        "payload": df.iloc[0]["payload"],
        "fidelity": df.iloc[0]["fidelity"],
        "predicted_label": predicted_label,
    }

# ====== Visualization TSNE/UMAP ======
@app.get("/visualize/{method}")
def visualize(method: str, model_name: str = Query("logreg")):
    method_clean = method.replace('"', '').strip().upper()
    if method_clean not in ["TSNE", "UMAP"]:
        return {"error": f"Unknown method '{method}'"}
    engine = create_engine("postgresql+psycopg2://postgres:100317@db/fakenewsdb")
    query = text("""
        SELECT p.news_id, p.x, p.y, n.text, l.label, l.predicted_label
        FROM ProjectionPoint p
        JOIN NewsItem n ON p.news_id = n.id
        JOIN Label l ON n.id = l.news_id
        JOIN Embedding e ON p.news_id = e.news_id
        WHERE UPPER(TRIM(BOTH '"' FROM p.method)) = :method
         AND e.model_id = :model
    """)
    df_coords = pd.read_sql(query, engine, params={"method": method_clean, "model": model_name})
    if df_coords.empty:
        return {"ids": [], "points": [], "labels": [], "predicted_labels": []}
    return {
        "model": model_name,
        "ids": df_coords["news_id"].tolist(),
        "points": df_coords[["x", "y"]].values.tolist(),
        "labels": df_coords["label"].tolist(),
        "predicted_labels": df_coords["predicted_label"].tolist()
    }

# ====== Roc Curve and Confusion Matrix ======
@app.get("/plots/{plot_name}")
def get_plot(plot_name: str, model_name: str = Query("logreg")):
    # Перевіряємо, що така модель є
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    # ❗ Додана перевірка готовності
    status = model_status.get(model_name)
    if not status or not status.get("ready"):
        raise HTTPException(
            status_code=409,  # Conflict
            detail=f"Plots not ready yet for model '{model_name}'. Training still in progress."
        )

    plot_dir = Path("output") / model_name
    plot_path = plot_dir / plot_name

    if plot_name not in {"confusion_matrix.png", "roc_curve.svg"}:
        raise HTTPException(status_code=404, detail="Plot not available.")

    if not plot_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Plot '{plot_name}' not found for model '{model_name}'."
        )

    return FileResponse(plot_path)

@app.post("/register")
def register_user(req: RegisterRequest):
    existing = get_user_by_email(req.email)
    if existing:
        raise HTTPException(status_code=400, detail="Користувач з таким email вже існує")
    if req.role not in ["user", "researcher"]:
        raise HTTPException(status_code=400, detail="Недопустима роль користувача")
    user = create_user(req.role, req.name, req.email, req.org, req.password)
    return {"status": "registered", "user": user}


@app.post("/login")
def login_user(req: LoginRequest):
    user = verify_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Невірний email або пароль")
    return {"status": "authenticated", "user": user}


def get_current_user():
    # Ця функція повинна повертати словник з user_id та role
    # Наприклад: {"id": 2, "role": "researcher"}
    # Для тестування можна повернути хардкод, але пам'ятайте про безпеку
    return {"id": 2, "role": "researcher"} # Тимчасова заглушка для тестування


@app.post("/feedback")
async def submit_feedback(
    type: str = Form(...),
    payload: str = Form(...), # payload тепер як str для Form data
    news_id: Optional[int] = Form(None),
    files: Optional[List[UploadFile]] = File(None), # Додаткові файли
    user: dict = Depends(get_current_user)
):
    """
    Приймає зміни від фронтенду (параметри або dataset) та зберігає у Feedback.
    """
    if user.get("role") != "researcher": # Перевірка ролі
        raise HTTPException(status_code=403, detail="Доступ заборонено. Тільки дослідники можуть надсилати зворотний зв'язок.")

    if type not in ["parameter_change", "dataset_update"]:
        raise HTTPException(status_code=400, detail="Invalid feedback type")

    parsed_payload = json.loads(payload) # Парсимо JSON рядок

    # Обробка нових файлів датасету, якщо вони є
    if type == "dataset_update" and files:
        results = []
        for file in files:
            try:
                from tempfile import NamedTemporaryFile
                with NamedTemporaryFile(delete=False, suffix=".tsv") as tmp:
                    tmp.write(await file.read())
                    tmp_path = tmp.name

                news_df, label_df = preprocess_liar_dataset(tmp_path, is_train=True)
                save_news_and_labels(news_df, label_df)

                results.append({
                    "filename": file.filename,
                    "rows_added": len(news_df)
                })
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        parsed_payload["dataset_upload_results"] = results # Додаємо результати завантаження до payload

    feedback_id = save_feedback(user["id"], news_id, type, parsed_payload)
    return {"status": "saved", "feedback_id": feedback_id, "payload_received": parsed_payload}
