import sqlite3
from config import DB_PATH


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_key   TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def save_history(conn: sqlite3.Connection, user_key: str, question: str, response: str):
    conn.execute(
        "INSERT INTO history (user_key, role, content) VALUES (?, ?, ?)",
        (user_key, "user", question),
    )
    conn.execute(
        "INSERT INTO history (user_key, role, content) VALUES (?, ?, ?)",
        (user_key, "assistant", response),
    )
    # Mantém as últimas 10 mensagens (5 trocas) por usuário
    conn.execute("""
        DELETE FROM history WHERE user_key = ? AND id NOT IN (
            SELECT id FROM history WHERE user_key = ? ORDER BY id DESC LIMIT 10
        )
    """, (user_key, user_key))
    conn.commit()


def load_history(conn: sqlite3.Connection, user_key: str) -> str:
    rows = conn.execute(
        "SELECT role, content FROM history WHERE user_key = ? ORDER BY id DESC LIMIT 10",
        (user_key,),
    ).fetchall()
    rows.reverse()
    return "\n".join(f"{role.capitalize()}: {content}" for role, content in rows)


def get_messages(conn: sqlite3.Connection, user_key: str) -> list[dict]:
    rows = conn.execute(
        "SELECT role, content FROM history WHERE user_key = ? ORDER BY id",
        (user_key,),
    ).fetchall()
    return [{"role": role, "content": content} for role, content in rows]


def clear_user_history(conn: sqlite3.Connection, user_key: str):
    conn.execute("DELETE FROM history WHERE user_key = ?", (user_key,))
    conn.commit()
