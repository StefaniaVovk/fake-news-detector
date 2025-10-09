import os
import time
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import json
from sqlalchemy import create_engine

def get_connection():
    """Підключення до PostgreSQL з повторними спробами."""
    while True:
        try:
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "db"),
                port=os.getenv("DB_PORT", "5432"),
                database=os.getenv("DB_NAME", "fakenewsdb"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "100317"),
            )
            print("✅ Підключення до БД успішне!")
            return conn
        except psycopg2.OperationalError as e:
            print("❌ БД ще не готова, повтор через 2 сек...", e)
            time.sleep(2)


def save_news_and_labels(news_df, label_df):
    """Збереження NewsItem + Label."""
    conn = get_connection()
    with conn.cursor() as cur:
        ids = []

        # 🔹 зберігаємо NewsItem
        for _, row in news_df.iterrows():
            published_at = row["published_at"]
            if pd.isna(published_at):
                published_at = None  # PostgreSQL NULL
            cur.execute(
                """
                INSERT INTO NewsItem (title, text, source, published_at, url, lang)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    row["title"],
                    row["text"],
                    row["source"],
                    published_at,
                    row["url"],
                    row["lang"],
                ),
            )
            ids.append(cur.fetchone()[0])

        # 🔹 зберігаємо Label
        for i, row in enumerate(label_df.itertuples()):
            timestamp = row.timestamp
            if pd.isna(timestamp):
                timestamp = None
            cur.execute(
                """
                INSERT INTO Label (news_id, label, annotator, confidence, timestamp, predicted_label)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    ids[i],
                    row.label,
                    row.annotator,
                    row.confidence,
                    timestamp,
                    None  # спочатку прогноз відсутній
                ),
            )

        conn.commit()
        print(f"💾 {len(ids)} новин та міток збережено.")
    conn.close()


def save_embeddings_bulk(news_ids, embeddings, model_id="sentence-bert"):
    
    conn = get_connection()
    with conn.cursor() as cur:
        for i, emb in enumerate(embeddings):
            cur.execute(
                """
                INSERT INTO Embedding (news_id, model_id, dim, vector_ref)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (news_id) DO UPDATE
                SET dim = EXCLUDED.dim, vector_ref = EXCLUDED.vector_ref;
                """,
                (news_ids[i], model_id, len(emb), emb.tolist())
            )
        conn.commit()
    conn.close()


def save_predicted_label(news_id, predicted_label, confidence):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE Label
            SET predicted_label = %s,
                confidence = %s
            WHERE news_id = %s
            """,
            (predicted_label, float(confidence), news_id)
        )
        conn.commit()
    conn.close()
    print(f"💾 Збережено прогноз для news_id={news_id}: {predicted_label} (confidence={confidence})")


def save_explanation(news_id, method, payload, fidelity=None):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO Explanation (news_id, method, payload, fidelity)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (news_id, method, json.dumps(payload), fidelity)
        )
        exp_id = cur.fetchone()[0]
        conn.commit()
    conn.close()
    try:
       print(f"💾 Explanation {method} для news_id={news_id} збережено (id={exp_id}).")
    except Exception as e:
        print(f"⚠️ Не вдалося зберегти пояснення: {e}")
    return exp_id


def save_projection_points(news_ids, method, coords, meta=None):
    """Збереження точок візуалізації (UMAP/t-SNE)."""
    conn = get_connection()
    with conn.cursor() as cur:
        for i, nid in enumerate(news_ids):
            cur.execute(
                """
                INSERT INTO ProjectionPoint (news_id, method, x, y, meta)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (news_id, method) DO UPDATE
                SET x = EXCLUDED.x, y = EXCLUDED.y, meta = EXCLUDED.meta;
                """,
                (
                    nid,
                    method,
                    float(coords[i][0]),
                    float(coords[i][1]),
                    json.dumps(meta) if meta else None,
                ),
            )
        conn.commit()
    conn.close()
    print(f"💾 ProjectionPoints ({method}) збережено: {len(news_ids)} записів.")

def load_all_texts():
    engine = create_engine("postgresql+psycopg2://postgres:100317@db/fakenewsdb")
    df = pd.read_sql(
        "SELECT n.text FROM NewsItem n JOIN Label l ON n.id = l.news_id WHERE l.predicted_label IS NULL;",
        engine
    )
    return df["text"].tolist() if not df.empty else []

def load_all_labels():
    engine = create_engine("postgresql+psycopg2://postgres:100317@db/fakenewsdb")
    df = pd.read_sql(
        "SELECT l.label FROM NewsItem n JOIN Label l ON n.id = l.news_id WHERE l.predicted_label IS NULL;",
        engine
    )
    return df["label"].tolist() if not df.empty else []

def load_all_news_ids():
    engine = create_engine("postgresql+psycopg2://postgres:100317@db/fakenewsdb")
    df = pd.read_sql(
        "SELECT n.id FROM NewsItem n JOIN Label l ON n.id = l.news_id WHERE l.predicted_label IS NULL;",
        engine
    )
    return df["id"].tolist() if not df.empty else []
