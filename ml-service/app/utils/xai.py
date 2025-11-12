# xai.py
import shap
from captum.attr import IntegratedGradients
from lime.lime_text import LimeTextExplainer
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import joblib


class ExplainableModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        if model_name == "logreg":
            # Завантажуємо треновану логрег-модель і енкодер
            self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.classifier = joblib.load("output/logreg/logreg.pkl")
        else:
            # Інші моделі (BERT)
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased").to(self.device)

    def explain_shap(self, text: str):
        if self.model_name == "logreg":
            # --- 1️⃣ Отримуємо embedding для тексту
            emb = self.embedder.encode([text])

            # --- 2️⃣ Формуємо background для SHAP (мінімальний контекст)
            background = self.embedder.encode([" "])

            # --- 3️⃣ Використовуємо LinearExplainer (інакше KernelExplainer як fallback)
            try:
                explainer = shap.LinearExplainer(self.classifier, background)
                shap_values = explainer.shap_values(emb)
            except Exception as e:
                print(f"⚠️ LinearExplainer failed ({e}), falling back to KernelExplainer")
                explainer = shap.KernelExplainer(self.classifier.predict_proba, background)
                shap_values = explainer.shap_values(emb, nsamples=100)

            # --- 4️⃣ Підтримка двох класів
            # LogisticRegression зазвичай має shap_values = [для класу0, для класу1]
            if isinstance(shap_values, list):
                shap_class0 = np.array(shap_values[0]).flatten().tolist()
                shap_class1 = np.array(shap_values[1]).flatten().tolist()
            else:
                shap_class0 = np.array(shap_values).flatten().tolist()
                shap_class1 = [-v for v in shap_class0]  # симетрично

            pred_class = int(np.argmax(self.classifier.predict_proba(emb)))
            scores = shap_class1 if pred_class == 1 else shap_class0

            # --- 5️⃣ Повертаємо пояснення без обрізання (повний вектор)
            return {
                "method": "SHAP",
                "tokens": None,  # для логрег токени не відомі
                "scores": scores
            }

        else:
            # --- інші моделі (BERT, DistilBERT)
            def predict_proba(texts):
                toks = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(self.device)
                with torch.no_grad():
                    return torch.softmax(self.model(**toks).logits, dim=-1).cpu().numpy()

            masker = shap.maskers.Text(self.tokenizer)
            explainer = shap.Explainer(predict_proba, masker)
            sv = explainer([text])

            scores = sv.values[0][np.argmax(sv.output[0])]
            tokens = sv.data[0]
            return {"method": "SHAP", "tokens": tokens, "scores": scores.tolist()}

    # --- Integrated Gradients (лише BERT)
    def explain_ig(self, text: str):
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        ig = IntegratedGradients(lambda inputs_embeds, attn_mask: self.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask).logits[:, 1])
        emb = self.model.get_input_embeddings()(enc["input_ids"])
        baseline = torch.zeros_like(emb)
        attrs = ig.attribute(emb, baselines=baseline, additional_forward_args=(enc["attention_mask"],))
        scores = attrs.norm(p=2, dim=-1).detach().cpu().numpy()[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        return {"method": "IG", "tokens": tokens[:15], "scores": scores[:15]}

    # --- LIME (лише BERT)
    def explain_lime(self, text: str):
        class_names = ["fake", "real"]
        def predict_proba(texts):
            toks = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                return torch.softmax(self.model(**toks).logits, dim=-1).cpu().numpy()
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(text, predict_proba, num_features=10)
        tokens, scores = zip(*exp.as_list())
        return {"method": "LIME", "tokens": tokens, "scores": scores}
