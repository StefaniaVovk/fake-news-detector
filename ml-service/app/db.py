import os
import time
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import json
from sqlalchemy import create_engine, text
import bcrypt
from typing import List

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
            # ids.append(cur.fetchone()[0])
            inserted_id = cur.fetchone()
            if inserted_id:
                ids.append(inserted_id[0])
            else:
                cur.execute("SELECT id FROM NewsItem WHERE text = %s", (row["text"],))
                existing_id = cur.fetchone()
                if existing_id:
                    ids.append(existing_id[0])

        # 🔹 зберігаємо Label
        for i, row in enumerate(label_df.itertuples()):

            if i >= len(ids):
                continue

            timestamp = row.timestamp
            if pd.isna(timestamp):
                timestamp = None

            label_value = int(row.label)

            cur.execute(
                "SELECT COUNT(*) FROM Label WHERE news_id = %s",
                (ids[i],)
            )
            label_exists = cur.fetchone()[0] > 0

            if not label_exists:
                cur.execute(
                    """
                    INSERT INTO Label (news_id, label, annotator, confidence, timestamp, predicted_label)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        ids[i],
                        label_value,
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
    method = str(method).replace('"', '').strip().upper()
    
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
    # Завантажуємо всі тексти, незалежно від того, чи є для них прогнози
    df = pd.read_sql(
        "SELECT n.id, n.text, l.label FROM NewsItem n JOIN Label l ON n.id = l.news_id;",
        engine
    )
    return df["id"].tolist(), df["text"].tolist(), df["label"].tolist() if not df.empty else ([], [], [])

def create_user(role, name, email, org, password):
    """Реєстрація користувача."""
    conn = get_connection()
    with conn.cursor() as cur:
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        cur.execute(
            """
            INSERT INTO Users (role, name, email, org, password_hash)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, role, name, email, org;
            """,
            (role, name, email, org, password_hash)
        )
        user = cur.fetchone()
        conn.commit()
    conn.close()
    return {"id": user[0], "role": user[1], "name": user[2], "email": user[3], "org": user[4]}


def get_user_by_email(email):
    """Пошук користувача за email."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute('SELECT id, role, name, email, org, password_hash FROM Users WHERE email = %s;', (email,))
        user = cur.fetchone()
    conn.close()
    if not user:
        return None
    return {
        "id": user[0],
        "role": user[1],
        "name": user[2],
        "email": user[3],
        "org": user[4],
        "password_hash": user[5]
    }


def verify_user(email, password):
    """Перевірка логіну."""
    user = get_user_by_email(email)
    if not user:
        return None
    if bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        return {k: v for k, v in user.items() if k != "password_hash"}
    return None

def save_feedback(user_id, news_id, change_type, payload):
    """
    Зберігає запис про зміну в Feedback.
    change_type: 'parameter_change' або 'dataset_update'
    payload: dict зі змінами
    """
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO Feedback (user_id, news_id, type, payload)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (user_id, news_id, change_type, json.dumps(payload))
        )
        feedback_id = cur.fetchone()[0]
        conn.commit()
    conn.close()
    print(f"💾 Feedback запис збережено (id={feedback_id})")
    return feedback_id

def get_unapplied_feedback_params(model_name: str):
    """
    Отримує незастосовані записи зворотного зв'язку типу 'parameter_change'
    для конкретної моделі.
    """
    engine = create_engine("postgresql+psycopg2://postgres:100317@db/fakenewsdb")
    query = text("""
        SELECT id, payload, created_at
        FROM Feedback
        WHERE type = 'parameter_change' AND applied = FALSE
        AND payload ->> 'model_name' = :model_name
        ORDER BY created_at ASC;
    """)
    df = pd.read_sql(query, engine, params={"model_name": model_name})
    return df.to_dict(orient="records") if not df.empty else []

def mark_feedback_applied(feedback_ids: List[int]):
    """
    Позначає записи зворотного зв'язку як застосовані.
    """
    if not feedback_ids:
        return

    conn = get_connection()
    with conn.cursor() as cur:
        # Використання UNNEST для передачі списку ID
        cur.execute(
            """
            UPDATE Feedback
            SET applied = TRUE
            WHERE id = ANY(%s::int[]);
            """,
            (feedback_ids,)
        )
        conn.commit()
    conn.close()
    print(f"✅ Позначено {len(feedback_ids)} записів Feedback як застосовані.")
