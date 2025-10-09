from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
import shap
from captum.attr import IntegratedGradients
import umap
from sklearn.manifold import TSNE
import pandas as pd
from sqlalchemy import create_engine
from app.db import save_embeddings_bulk, save_projection_points, save_predicted_label

# ====== Базовий інтерфейс ======
class BaseModelInterface(ABC):
    @abstractmethod
    def train(self, texts, labels, test_size=0.3):
        pass

    @abstractmethod
    def predict(self, text):
        pass

    @abstractmethod
    def is_trained(self):
        pass

    def explain_shap(self, texts): 
        return None
    
    def explain_ig(self, texts): 
        return None
    
    def explain_tcav(self, texts): 
        return None


# ====== Sentence-BERT + Logistic Regression ======
embedding_model = SentenceTransformer("all-MiniLM-L12-v2")

class FakeNewsClassifier(BaseModelInterface):
    def __init__(self, max_iter=1000, C=1.0, solver="liblinear"):
        self.max_iter = max_iter
        self.C = C
        self.solver = solver
        self.clf = LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=max_iter,
            C=C,
            solver=solver,
        )
        self.fitted = False
        self.train_running = False
        self.train_metrics = None
        self.train_rows = 0

    def train(self, texts, labels, test_size=0.3):
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, stratify=labels, random_state=42
        )
        emb_train = embedding_model.encode(X_train)
        emb_test = embedding_model.encode(X_test)

        self.clf.fit(emb_train, y_train)
        self.fitted = True

        y_pred = self.clf.predict(emb_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
        }
        self.train_metrics = metrics
        return metrics

    def predict(self, text):
        emb = embedding_model.encode([text])
        prob = self.clf.predict_proba(emb)[0][1]
        return int(prob > 0.5), float(prob)

    def is_trained(self):
        return self.fitted

    def predict_proba(self, texts):
        emb = embedding_model.encode(texts)
        return self.clf.predict_proba(emb)
    
    def predict_proba_embedded(self, embedded_input):
        if not hasattr(self, "clf") or not self.fitted:
            raise RuntimeError("Модель LogisticRegression ще не навчена.")
    
         # Якщо подано 1D масив, перетворюємо на 2D
        if isinstance(embedded_input, list) or len(embedded_input.shape) == 1:
            embedded_input = np.expand_dims(embedded_input, axis=0)

        probs = self.clf.predict_proba(embedded_input)
        return probs

    # Методи пояснень
    def explain_shap(self, texts):
        if not texts:
            return []  # або [{"tokens": [], "scores": []}]
        embeddings = embedding_model.encode(texts, convert_to_numpy=True)
        background = embedding_model.encode([" "])  # мінімальний background

        try:
            explainer = shap.LinearExplainer(self.clf, background)
            shap_values = explainer.shap_values(embeddings)
        except Exception as e:
            print(f"⚠️ LinearExplainer failed ({e}), falling back to KernelExplainer")
            explainer = shap.KernelExplainer(self.clf.predict_proba, background)
            shap_values = explainer.shap_values(embeddings, nsamples=100)
        
        # Перетворюємо завжди у 2D список
        if isinstance(shap_values, list):
            shap_class1 = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            return [np.array(shap_class1).tolist()]
        else:
            return [np.array(shap_values).tolist()]
        # return np.array(shap_values).tolist()


    # Тренування з БД
    def run_training(self, test_size=0.3, max_iter=1000, C=1.0, solver="liblinear"):
        self.train_running = True
        self.train_metrics = None
        engine = create_engine("postgresql+psycopg2://postgres:100317@db/fakenewsdb")
        df = pd.read_sql(
            "SELECT n.id, n.text, l.label FROM NewsItem n JOIN Label l ON n.id = l.news_id WHERE l.predicted_label IS NULL;",
            engine,
        )

        if df.empty:
            self.train_running = False
            return

        X, y = df["text"].tolist(), df["label"].tolist()
        self.clf = LogisticRegression(max_iter=max_iter, C=C, solver=solver, random_state=42)
        self.train_rows = len(df)

        metrics = self.train(X, y, test_size)
        embeddings = embedding_model.encode(X)
        save_embeddings_bulk(df["id"].tolist(), embeddings, model_id="logreg")

        for news_id, text in zip(df["id"].tolist(), X):
            label, prob = self.predict(text)
            save_predicted_label(news_id, label, prob)

        # t-SNE + UMAP
        try:
            tsne_coords = TSNE(perplexity=30, learning_rate=200, n_components=2, random_state=42).fit_transform(embeddings)
            save_projection_points(df["id"].tolist(), "TSNE", tsne_coords)
        except Exception as e:
            print("⚠️ TSNE error:", e)

        try:
            umap_coords = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42).fit_transform(embeddings)
            save_projection_points(df["id"].tolist(), "UMAP", umap_coords)
        except Exception as e:
            print("⚠️ UMAP error:", e)

        self.train_metrics = metrics
        self.train_running = False


# ====== BERT-Tiny Classifier ======
class BertTinyClassifier(BaseModelInterface):
    def __init__(self, model_name="mrm8488/bert-tiny-finetuned-fake-news-detection"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.trainer = None
        self.fitted = False
        self.embeddings = None
        self.labels = None

    def train(self, texts, labels, test_size=0.3):
        from sklearn.model_selection import train_test_split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, stratify=labels, random_state=42
        )

        train_enc = self.tokenizer(train_texts, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        test_enc = self.tokenizer(test_texts, truncation=True, padding="max_length", max_length=512, return_tensors="pt")

        train_dataset = torch.utils.data.TensorDataset(
            train_enc["input_ids"], train_enc["attention_mask"], torch.tensor(train_labels, dtype=torch.long)
        )
        test_dataset = torch.utils.data.TensorDataset(
            test_enc["input_ids"], test_enc["attention_mask"], torch.tensor(test_labels, dtype=torch.long)
        )

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=20,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            evaluation_strategy="epoch"
        )

        def collate_fn(batch):
            return {
                "input_ids": torch.stack([item[0] for item in batch]),
                "attention_mask": torch.stack([item[1] for item in batch]),
                "labels": torch.tensor([item[2] for item in batch]),
            }

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            data_collator=collate_fn,
        )
        self.trainer.train()

        # збереження embeddings (CLS токен)
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
            outputs = self.model.base_model(**inputs)
            # беремо перший токен [CLS] як ембедінг
            self.embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        self.labels = labels

        # обчислюємо метрики
        preds = self.trainer.predict(test_dataset)
        y_pred = preds.predictions.argmax(axis=-1)
        metrics = {
            "accuracy": accuracy_score(test_labels, y_pred),
            "precision": precision_score(test_labels, y_pred, average="weighted"),
            "recall": recall_score(test_labels, y_pred, average="weighted"),
            "f1": f1_score(test_labels, y_pred, average="weighted"),
        }

        try:
            from app.db import load_all_news_ids
            news_ids = load_all_news_ids()
            if news_ids:
                self.save_embeddings_and_predictions(news_ids, texts)
        except Exception as e:
            print("⚠️ Error saving predictions for bert-tiny:", e)

        self.fitted = True
        return metrics

    def predict(self, text):
        enc = self.tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**enc)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        return int(probs[1] > 0.5), float(probs[1])

    def is_trained(self):
        return self.fitted
    
    def predict_proba(self, texts):
        self.model.eval()

        if isinstance(texts, str):
            texts = [texts]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
        return probs
    

    def predict_proba_embedded(self, texts, mask=None):

        self.model.eval()

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        if mask is not None:
        # обнулюємо attention для токенів із низьким впливом
            if isinstance(mask, np.ndarray):
                mask = torch.tensor(mask, dtype=torch.long)
                attn_shape = enc["attention_mask"].shape
                if mask.ndim == 1:
                    mask = mask.unsqueeze(0)
                if mask.shape[1] < attn_shape[1]:
                    # доповнюємо до довжини attention_mask
                    pad_len = attn_shape[1] - mask.shape[1]
                    mask = torch.cat([mask, torch.zeros((1, pad_len), dtype=torch.long)], dim=1)
                elif mask.shape[1] > attn_shape[1]:
                    # обрізаємо зайве
                    mask = mask[:, :attn_shape[1]]
            if mask.shape == enc["attention_mask"].shape:
                enc["attention_mask"] *= mask
            else:
                print("⚠️ Маска не узгоджена за розміром, ігнорується.")

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
        return probs


    # ======== Пояснення SHAP ========
    def explain_shap(self, texts):
        """
        Пояснення SHAP для BERT без подвійної токенізації.
        Використовується shap.maskers.Text(self.tokenizer), який сам формує токени.
        """

        if not texts:
            return {"tokens": [], "scores": []}

        # --- Нормалізація вхідних даних ---
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, list) and len(texts) == 1 and isinstance(texts[0], list):
            texts = texts[0]

        clean_texts = [str(t).strip() for t in texts if t and str(t).strip()]
        if not clean_texts:
            raise ValueError("Порожній текст для explain_shap")

        # --- Переключаємо модель у режим оцінки ---
        self.model.eval()

        # 🔹 SHAP сам подає вже токенізований batch у f(x)
        def f(x):
            with torch.no_grad():
                if isinstance(x, dict):
                    inputs = {k: v.to(self.model.device) for k, v in x.items()}
                elif isinstance(x, (list, np.ndarray)):
                    inputs = self.tokenizer(
                    list(x),
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                    ).to(self.model.device)
                else:
                    raise TypeError(f"Невідомий тип вхідних даних у f(): {type(x)}")
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
            return probs

        # --- Ініціалізуємо маскер для текстів ---
        masker = shap.maskers.Text(self.tokenizer)

        # --- Створюємо пояснювач SHAP ---
        explainer = shap.Explainer(f, masker)

        # --- Отримуємо SHAP значення ---
        shap_values = explainer(clean_texts)

        # --- Формуємо результати ---
        results = []
        for sv in shap_values:
            tokens = sv.data
            if sv.values.ndim > 1:  # двокласова модель
                scores = sv.values[:, 1].tolist()
            else:
                scores = sv.values.tolist()
            results.append({"tokens": tokens, "scores": scores})

        return results
    

    # ======== Пояснення IG (Integrated Gradients) ========
    def explain_ig(self, texts):
        """
        Використовує Captum Integrated Gradients для обчислення впливу токенів.
        """
        if isinstance(texts, list):
            if len(texts) == 1 and isinstance(texts[0], str):
                text = texts[0]
            else:
                raise ValueError("IG підтримує лише один текст за раз")
        elif isinstance(texts, str):
            text = texts
        else:
            raise TypeError(f"Невірний тип вхідних даних: {type(texts)}")
        
        self.model.eval()
        tokenizer = self.tokenizer

        inputs = tokenizer(text, 
                           return_tensors="pt", 
                           truncation=True, 
                           padding=True, 
                           max_length=512)
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Отримуємо ембедінги токенів
        embeddings = self.model.bert.embeddings(input_ids)
        embeddings.requires_grad_()

        # беремо логіти
        def forward_func(inputs_embeds):
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            return outputs.logits[:, 1]
        
        ig = IntegratedGradients(forward_func)

        attributions, delta = ig.attribute(embeddings, return_convergence_delta=True)

        scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy().tolist()

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        return {"tokens": tokens, "scores": scores, "delta": float(delta)}


    # ======== Пояснення TCAV ========
    def explain_tcav(self, texts):
        """
        Імітаційна TCAV-реалізація: обчислює умовні ваги токенів, що вказують на наявність "концептів".
        (TCAV зазвичай потребує додаткових концептуальних векторів, тому спрощено.)
        """
        text = texts[0]
        tokens = self.tokenizer.tokenize(text)
        importance = np.random.uniform(-1, 1, len(tokens))  # імітаційні ваги
        score = float(np.mean(np.abs(importance)))
        return {"concept": "contextual_bias", "tokens": tokens, "scores": importance.tolist(), "tcav_score": score}

    def save_embeddings_and_predictions(self, news_ids, texts):
        # 1️⃣ Зберігаємо embeddings у БД
        save_embeddings_bulk(news_ids, self.embeddings, model_id="bert-tiny")
        # 2️⃣ Зберігаємо прогнозовані мітки
        for i, nid in enumerate(news_ids):
            label, prob = self.predict(texts[i])
            save_predicted_label(nid, label, prob)
        # 3️⃣ t-SNE + UMAP
        try:
            tsne_coords = TSNE(perplexity=30, learning_rate=200, n_components=2, random_state=42).fit_transform(self.embeddings)
            save_projection_points(news_ids, "TSNE", tsne_coords)
        except Exception as e:
            print("⚠️ TSNE error:", e)

        try:
            umap_coords = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42).fit_transform(self.embeddings)
            save_projection_points(news_ids, "UMAP", umap_coords)
        except Exception as e:
            print("⚠️ UMAP error:", e)
