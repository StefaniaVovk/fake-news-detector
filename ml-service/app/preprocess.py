import pandas as pd
import re
import unicodedata
from langdetect import detect
from datetime import datetime
from typing import Tuple

def clean_text(text: str) -> str:
    """Нормалізація тексту."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_lang_safe(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"

def preprocess_liar_dataset(filepath: str, is_train=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Обробка LIAR train/test TSV датасету.
    Повертає: (news_df, label_df)
    """
    # 🔹 1. Завантаження TSV
    df = pd.read_csv(filepath, sep='\t', header=None, names=[
        "id", "label", "statement", "subjects", "speaker", "speaker_job",
        "state_info", "party", "barely_true_counts", "false_counts",
        "half_true_counts", "mostly_true_counts", "pants_on_fire_counts",
        "context"
    ])

    # 🔹 2. Очищення тексту
    df["clean_statement"] = df["statement"].apply(clean_text)
    df["lang"] = df["clean_statement"].apply(detect_lang_safe)

    # 🔹 3. Мапінг лейблів (до 0/1)
    label_map = {
        "pants-fire": 0,
        "false": 0,
        "barely-true": 0,
        "half-true": 0,
        "mostly-true": 1,
        "true": 1,
    }
    df["binary_label"] = df["label"].map(label_map)

    # 🔹 4. Формуємо DataFrame для NewsItem
    df["title"] = df["statement"].apply(lambda x: x[:80] + "..." if isinstance(x, str) and len(x) > 80 else x)
    df["source"] = df.apply(lambda r: f"{r['party']} - {r['speaker']}" if pd.notna(r['party']) and pd.notna(r['speaker']) else r['speaker'], axis=1)

    news_df = df[["id", "title", "statement", "source", "lang"]].rename(
        columns={
            "id": "id",
            "statement": "text",
        }
    )
    news_df["published_at"] = None
    news_df["url"] = None

    # 🔹 5. Формуємо DataFrame для Label
    label_df = pd.DataFrame({
        "news_id": df["id"],
        "label": df["binary_label"],
        "annotator": df["speaker"].fillna("auto"),
        "confidence": 1.0,
        "timestamp": datetime.utcnow(),
        "predicted_label": None
    })

    # 🔹 6. Видаляємо записи без міток
    news_df = news_df[df["binary_label"].notna()].reset_index(drop=True)
    label_df = label_df[df["binary_label"].notna()].reset_index(drop=True)

    # 🔹 7. Балансування класів
    fake_df = news_df[df["id"].isin(df[df["binary_label"] == 0]["id"])]
    real_df = news_df[df["id"].isin(df[df["binary_label"] == 1]["id"])]
    min_len = min(len(fake_df), len(real_df))
    balanced_ids = pd.concat([
        fake_df.sample(min_len, random_state=42),
        real_df.sample(min_len, random_state=42)
    ])["id"]

    news_df = news_df[news_df["id"].isin(balanced_ids)]
    label_df = label_df[label_df["news_id"].isin(balanced_ids)]

    print(f"✅ Оброблено {len(news_df)} записів ({'train' if is_train else 'test'}). Баланс 1:1 між класами.")
    return news_df, label_df

