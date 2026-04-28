import os
import json
import sqlite3
import argparse

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CHROMA_PATH = "chroma"
TOPICS_PATH = "chroma/topics.json"
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DB_PATH = os.getenv("HISTORY_DB", "history.db")

# Cosine similarity: score maior = mais similar (ChromaDB retorna valores entre 0 e 1)
HYBRID_THRESHOLD = 0.5   # abaixo disso, ignora o filtro e faz busca global
NO_INFO_THRESHOLD = 0.3  # abaixo disso, não há informação relevante

ROUTER_PROMPT = """Você é um classificador de perguntas sobre Computação Gráfica.
Classifique a pergunta abaixo no tópico mais adequado da lista.
Responda APENAS com o texto exato do tópico (cópia da lista), sem explicações.
Se não encaixar em nenhum, responda "Geral".

Tópicos disponíveis:
{topics}

Pergunta: {question}

Tópico:"""

ANSWER_PROMPT = """Você é uma IA monitora de Computação Gráfica especializada em {topic}.
Responda a pergunta com base no contexto fornecido.
NÃO utilize informações de fora do contexto. A resposta será impressa em terminal, NÃO use formatação especial.

Contexto:
{context}

Pergunta: {question}

Resposta:"""

RESUME_PROMPT = """Com base nesse histórico: [{history}], e nesse input: [{question}], crie APENAS um input resumindo o que o usuário deseja saber para ser utilizado em um agente de LLM. NUNCA responda a pergunta do usuário."""


# --- Histórico com SQLite ---

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
    # Mantém apenas as últimas 10 mensagens (5 trocas) por usuário
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


# --- Router semântico ---

def route_question(question: str, topics: list[str], model: ChatOllama) -> str:
    result = (ChatPromptTemplate.from_template(ROUTER_PROMPT) | model).invoke({
        "topics": "\n".join(f"- {t}" for t in topics),
        "question": question,
    }).content.strip()
    return result if result in topics else "Geral"


# --- Busca híbrida ---

def hybrid_search(db: Chroma, query: str, topic: str, k: int = 5):
    filtered = db.similarity_search_with_relevance_scores(
        query, k=k, filter={"topic": topic}
    )
    if filtered and filtered[0][1] >= HYBRID_THRESHOLD:
        return filtered, "filtrado"
    return db.similarity_search_with_relevance_scores(query, k=k), "global"


# --- Main ---

def main(user_key: str, question: str):
    conn = init_db()
    model = ChatOllama(model=CHAT_MODEL)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    with open(TOPICS_PATH, encoding="utf-8") as f:
        topics: list[str] = json.load(f)["topics"]

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Enriquece a pergunta com o contexto do histórico, se houver
    history = load_history(conn, user_key)
    if history:
        query = (ChatPromptTemplate.from_template(RESUME_PROMPT) | model).invoke(
            {"history": history, "question": question}
        ).content.strip()
    else:
        query = question

    # Detecta tópico e executa busca híbrida
    topic = route_question(query, topics, ChatOllama(model=CHAT_MODEL, temperature=0))
    results, search_type = hybrid_search(db, query, topic)

    if not results or results[0][1] < NO_INFO_THRESHOLD:
        print("Lamento, não possuo informações sobre esse assunto.")
        conn.close()
        return

    context = "\n\n---\n\n".join(doc.page_content for doc, _ in results)
    sources = list(set(doc.metadata.get("source", "") for doc, _ in results))

    response = (ChatPromptTemplate.from_template(ANSWER_PROMPT) | model).invoke(
        {"topic": topic, "context": context, "question": query}
    )

    save_history(conn, user_key, question, response.content)
    conn.close()

    print(f"[Tópico: {topic} | Busca: {search_type}]")
    print(response.content)
    if sources:
        print("\nFontes:\n" + "\n".join(sources))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("user_key")
    parser.add_argument("question")
    args = parser.parse_args()
    main(user_key=args.user_key, question=args.question)
