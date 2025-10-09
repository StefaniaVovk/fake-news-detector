from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from typing import List
import pandas as pd
from app.preprocess import clean_text, preprocess_dataset
from app.db import save_news_and_labels, save_explanation
from app.model import FakeNewsClassifier, BertTinyClassifier
from sqlalchemy import create_engine, text
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import random

app = FastAPI(title="FakeNews ML Service")

# ✅ Увімкнення CORS
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

model_status = {name: {"running": False, "metrics": None, "ready": False} for name in models.keys()}

def get_model(name: str):
    if name not in models:
        raise HTTPException(status_code=400, detail=f"Unknown model '{name}'")
    return models[name]

# ====== Pydantic Models ======
class PredictRequest(BaseModel):
    news_text: str
    model_name: str = Field("logreg", description="Назва моделі: logreg або bert-tiny")
    
    class Config:
        protected_namespaces = ()


class AnalyzeRequest(BaseModel):
    model_name: str = Field("logreg", description="Назва моделі: logreg або bert-tiny")
    test_size: float = Field(0.3, alias="testSize")
    max_iter: int = 10
    C: float = 1.0
    solver: str = "liblinear"

    class Config:
        allow_population_by_field_name = True
        protected_namespaces = ()


# ====== Health ======
@app.get("/health")
def health():
    return {"status": "ok"}


# ====== Preprocess ======
@app.post("/preprocess")
async def preprocess(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            df = pd.read_csv(file.file)
            news_df, label_df = preprocess_dataset(df, file.filename)
            save_news_and_labels(news_df, label_df)
            results.append({"filename": file.filename, "rows": len(df)})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    return {"status": "preprocessed", "files": results}


# ====== Training ======
@app.post("/analyze")
def analyze_all(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    ml_model = get_model(request.model_name)
    status = model_status.get(request.model_name)

    if status["running"]:
        return {"status": "already_running", "model": request.model_name}

    def background_train():
        model_status[request.model_name]["running"] = True
        try:
            # --- FakeNewsClassifier ---
            if hasattr(ml_model, "run_training"):
                ml_model.run_training(
                    test_size=getattr(request, "test_size", 0.3),
                    max_iter=getattr(request, "max_iter", 1000),
                    C=getattr(request, "C", 1.0),
                    solver=getattr(request, "solver", "liblinear")
                )
                metrics = getattr(ml_model, "train_metrics", None)

            # --- BertTinyClassifier ---
            elif hasattr(ml_model, "train"):
                from app.db import load_all_texts, load_all_labels, load_all_news_ids

                texts = load_all_texts()
                labels = load_all_labels()
                news_ids = load_all_news_ids()

                if not texts or not labels or not news_ids:
                    raise HTTPException(status_code=400, detail="Немає даних для тренування")

                metrics = ml_model.train(texts=texts, 
                                         labels=labels, 
                                         test_size=getattr(request, "test_size", 0.3))

            else:
                raise HTTPException(status_code=400, 
                                    detail=f"Модель '{request.model_name}' не підтримує тренування")

            # Оновлюємо статус
            model_status[request.model_name]["metrics"] = metrics
            model_status[request.model_name]["ready"] = True

            # --- ✅ Зберігаємо оновлену модель у пам'яті ---
            models[request.model_name] = ml_model

        finally:
            model_status[request.model_name]["running"] = False

    # Запускаємо тренування у фоні
    background_tasks.add_task(background_train)

    return {
        "status": "training_started",
        "model": request.model_name,
        "params": {
            "test_size": getattr(request, "test_size", None),
            "max_iter": getattr(request, "max_iter", None),
            "C": getattr(request, "C", None),
            "solver": getattr(request, "solver", None),
        }
    }

@app.get("/analyze/status")
def analyze_status(model_name: str = Query("logreg", description="Назва моделі")):
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


@app.get("/random_predict")
def random_predict(model_name: str = Query("logreg", description="Назва моделі")):
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
        predicted_label = "real" if row.predicted_label == 0 else "fake"
        true_label = "real" if row.label is False else "fake"

        explanations_created = []
        supported_methods = {
            "logreg": ["SHAP"],
            "bert-tiny": ["SHAP", "IG", "TCAV"]
        }

        for method in supported_methods.get(model_name, []):
            try:

                if method == "SHAP":
                    if model_name == "bert-tiny":
                        shap_result = ml_model.explain_shap(text_item)
                    else: 
                        shap_result = ml_model.explain_shap([text_item])

                    # беремо лише scores для fidelity
                    if isinstance(shap_result, list) and "scores" in shap_result[0]:
                        values = shap_result[0]["scores"]
                    else:
                        values = shap_result

                elif method == "IG":
                    ig_result = ml_model.explain_ig(text_item)
                    values = ig_result["scores"]
                elif method == "TCAV":
                    values = ml_model.explain_tcav([text_item])
                else:
                    continue

                # --- Перевірка даних ---
                if not values or len(values) == 0 :
                    print(f"⚠️ {method} повернув порожні значення")
                    fidelity = None
                else:
                    # --- Обчислення fidelity ---
                    p_orig = ml_model.predict_proba([text_item])[0][1]
                    weights = np.array(values)
                    threshold = np.percentile(np.abs(weights), 70)
                    modified_input = np.copy(weights)
                    modified_input[np.abs(weights) < threshold] = 0

                    try:
                        if model_name == "bert-tiny" and hasattr(ml_model, "predict_proba_embedded"):
                            mask = np.ones_like(weights, dtype=int)
                            mask[np.abs(weights) < threshold] = 0
                            p_expl = ml_model.predict_proba_embedded([text_item], mask=mask)[0][1]
                        elif hasattr(ml_model, "predict_proba_embedded"):
                            p_expl = ml_model.predict_proba_embedded(modified_input)[0][1]
                        else:
                            p_expl = p_orig
                    except Exception as e:
                        print(f"⚠️ predict_proba_embedded failed: {e}")
                        p_expl = p_orig

                    fidelity = float(1 - abs(p_orig - p_expl))

                # --- Зберігаємо у базу ---
                save_explanation(random_id, method, values, fidelity)
                explanations_created.append(method)

            except Exception as e:
                print(f"⚠️ Explanation for {method} failed: {e}")

        return {
            "news_id": random_id,
            "model": model_name,
            "text": text_item[:300] + "..." if len(text_item) > 300 else text_item,
            "prediction": {
                "predicted_label": predicted_label,
                "probability": float(row.confidence)
            },
            "true_label": true_label,
            "created_explanations": explanations_created
        }


# ====== Interpretability ======
@app.post("/interpret/{method}")
def interpret(method: str, news_id: int, model_name: str = Query("logreg")):
    engine = create_engine("postgresql+psycopg2://postgres:100317@db/fakenewsdb")

    query = text("""
        SELECT e.payload, e.fidelity
        FROM Explanation e
        JOIN Embedding em ON e.news_id = em.news_id
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
        raise HTTPException(status_code=404, detail="Пояснення не знайдено. Спочатку створіть його через /random_predict")

    return {
        "news_id": news_id,
        "model": model_name,
        "method": method.upper(),
        "payload": df.iloc[0]["payload"],
        "fidelity": df.iloc[0]["fidelity"],
    }


# ====== Visualization ======
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
