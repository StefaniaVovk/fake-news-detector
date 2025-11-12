# bert_tiny_classifier.py
import matplotlib
matplotlib.use('Agg')

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from app.db import (
    load_all_texts, save_embeddings_bulk, save_projection_points, save_predicted_label
)
from app.utils.projection import compute_tsne, compute_umap
from app.models.dataset import NewsDataset
import traceback

class BertTinyClassifier:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.trained = False
        self.train_metrics = None
        self.output_dir = Path.cwd() / "output" / "bert-tiny"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir: {self.output_dir}")

    def train(self, test_size=0.3):
        # --- 1️⃣ Завантаження даних
        news_ids, texts, labels = load_all_texts()

        if not texts:
            raise ValueError("Немає текстів для тренування")

        # --- 2️⃣ Поділ на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # --- 3️⃣ Формування датасетів
        train_dataset = NewsDataset(self.tokenizer, X_train, y_train)
        test_dataset = NewsDataset(self.tokenizer, X_test, y_test)

        # --- 4️⃣ Параметри навчання
        args = TrainingArguments(
            output_dir=str(self.output_dir),
            logging_dir=str(self.output_dir / "logs"),
            num_train_epochs=10, 
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=2e-5,
            logging_steps=100,
            evaluation_strategy="epoch",
            no_cuda=True,                 
            dataloader_num_workers=2,     
            disable_tqdm=False            # щоб бачити прогрес у логах
        )

        trainer = Trainer(
            model=self.model, 
            args=args, 
            train_dataset=train_dataset, 
            eval_dataset=test_dataset
        )
        trainer.train()
        self.trained = True

        # --- 5️⃣ Оцінка на тестових даних
        preds = trainer.predict(test_dataset)
        probs = torch.nn.functional.softmax(torch.tensor(preds.predictions), dim=1)[:, 1].numpy()
        y_pred = np.argmax(preds.predictions, axis=1)

        report = classification_report(y_test, y_pred, output_dict=True)
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

        # --- 6️⃣ Збереження графіків
        try:
            plt.figure(figsize=(6, 5))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix - DistilBERT")
            plt.savefig(self.output_dir / "confusion_matrix.png")
            plt.close()

            plt.figure()
            plt.plot(fpr, tpr, color="orange", lw=2, label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.title("ROC Curve")
            plt.legend()
            plt.savefig(self.output_dir / "roc_curve.svg")
            plt.close()
        except Exception as e:
            print(f"⚠️ Could not save visualizations: {e}")
            traceback.print_exc()

        # --- 7️⃣ Отримання embedding’ів для всіх текстів
        with torch.no_grad():
            inputs = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
            outputs = self.model.distilbert(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # --- 8️⃣ Збереження embedding’ів у базу
        save_embeddings_bulk(news_ids, embeddings, model_id="bert-tiny")

        # --- 9️⃣ Збереження TSNE/UMAP
        tsne_coords = compute_tsne(embeddings)
        umap_coords = compute_umap(embeddings)
        save_projection_points(news_ids, "TSNE", tsne_coords)
        save_projection_points(news_ids, "UMAP", umap_coords)

        # --- 🔟 Збереження прогнозів у базу (новий блок)
        try:
            for start in range(0, len(texts), 16):  # батчами, щоб не переповнити пам’ять
                batch_texts = texts[start:start + 16]
                toks = self.tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**toks)
                    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)
                    for nid, p, conf in zip(news_ids[start:start + 16], preds, probs[:, 1]):
                        save_predicted_label(nid, int(p), float(conf))
        except Exception as e:
            print(f"⚠️ Could not save predictions: {e}")
            traceback.print_exc()

        # --- 11️⃣ Метрики
        if "1" in report:
            key = "1"
        elif True in report:
            key = True
        else:
            key = list(report.keys())[0]

        self.train_metrics = {
            "accuracy": report.get("accuracy", 0.0),
            "precision": report.get(key, {}).get("precision", 0.0),
            "recall": report.get(key, {}).get("recall", 0.0),
            "f1": report.get(key, {}).get("f1-score", 0.0),
            "roc_auc": roc_auc
        }

    def is_trained(self):
        return self.trained
