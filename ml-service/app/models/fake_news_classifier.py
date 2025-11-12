import matplotlib
matplotlib.use('Agg')

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from app.db import save_embeddings_bulk, save_projection_points
from app.utils.projection import compute_tsne, compute_umap
import traceback

class FakeNewsClassifier:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.classifier = None
        self.is_fit = False
        self.train_metrics = None
        self.output_dir = Path.cwd() / "output" / "logreg"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir: {self.output_dir}")

    def run_training(self, test_size=0.3, max_iter=1000, C=5.0, solver="liblinear"):
        from app.db import load_all_texts

        news_ids, texts, labels = load_all_texts()
        if not texts:
            raise ValueError("Немає текстів для тренування")

        X = self.embedder.encode(texts, show_progress_bar=True)
        y = np.array(labels)

        unique, counts = np.unique(y, return_counts=True)
        print("📊 Кількість прикладів по класах:", dict(zip(unique, counts)))

        save_embeddings_bulk(news_ids, X, model_id="logreg")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        self.classifier = LogisticRegression(max_iter=max_iter, C=C, solver=solver, class_weight="balanced")
        self.classifier.fit(X_train, y_train)
        self.is_fit = True

        y_pred = self.classifier.predict(X_train)
        y_prob = self.classifier.predict_proba(X_train)[:, 1]

        y_pred_test = self.classifier.predict(X_test)

        report = classification_report(y_train, y_pred, output_dict=True)
        cm = confusion_matrix(y_train, y_pred)
        fpr, tpr, _ = roc_curve(y_train, y_prob)
        roc_auc = auc(fpr, tpr)

        try:
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix - Sentence-BERT + LogReg")
            plt.savefig(self.output_dir / "confusion_matrix.png")
            plt.close()
            print(f"💾 Збереження матриці помилок у {self.output_dir / 'confusion_matrix.png'}")
        except Exception as e:
            print(f"⚠️ Could not save confusion matrix: {e}")
            traceback.print_exc()

        try:
            plt.figure()
            plt.plot(fpr, tpr, color="orange", lw=2, label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.title("ROC Curve - Sentence-BERT + LogReg")
            plt.legend()
            plt.savefig(self.output_dir / "roc_curve.svg")
            plt.close()
            print(f"💾 Збереження roc-кривої у {self.output_dir / 'roc_curve.svg'}")
        except Exception as e:
            print(f"⚠️ Could not save confusion matrix: {e}")
            traceback.print_exc()

        from app.db import save_predicted_label
        y_all_pred = self.classifier.predict(X)
        y_all_prob = self.classifier.predict_proba(X)[:, 1]
        for news_id, pred, conf in zip(news_ids, y_all_pred, y_all_prob):
            save_predicted_label(news_id, int(pred), float(conf))

        # Зберігаємо проєкції
        tsne_coords = compute_tsne(X)
        umap_coords = compute_umap(X)
        save_projection_points(news_ids, "TSNE", tsne_coords)
        save_projection_points(news_ids, "UMAP", umap_coords)

        joblib.dump(self.classifier, self.output_dir / "logreg.pkl")

        if "1" in report:
            key = "1"
        elif True in report:
            key = True
        else:
            key = list(report.keys())[0]

        # Метрики з тестового набору даних
        test_report = classification_report(y_test, y_pred_test, output_dict=True)
        test_y_prob = self.classifier.predict_proba(X_test)[:, 1]
        test_fpr, test_tpr, _ = roc_curve(y_test, test_y_prob)
        test_roc_auc = auc(test_fpr, test_tpr)

        self.train_metrics = {
            "accuracy": report.get("accuracy", 0.0),
            "precision": report.get(key, {}).get("precision", 0.0),
            "recall": report.get(key, {}).get("recall", 0.0),
            "f1": report.get(key, {}).get("f1-score", 0.0),
            "roc_auc": roc_auc
        }
        return self.train_metrics

    def predict(self, text):
        if not self.is_fit:
            model_path = self.output_dir / "logreg.pkl"
            if model_path.exists():
                self.classifier = joblib.load(model_path)
                self.is_fit = True
            else:
                raise ValueError(f"Модель logreg ще не натренована і файл моделі не знайдено.")

        emb = self.embedder.encode([text])
        label = self.classifier.predict(emb)[0]
        prob = self.classifier.predict_proba(emb)[0][label]
        return int(label), float(prob)

    def is_trained(self):
        return self.is_fit or (self.output_dir / "logreg.pkl").exists()
