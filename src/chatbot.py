import os
import json
import argparse

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from config import (
    CHROMA_PATH, TOPICS_PATH, CHAT_MODEL, EMBED_MODEL,
    NO_INFO_THRESHOLD, GLOBAL_SEARCH,
)
from agents.history import init_db, load_history, save_history
from agents.router import route_question
from agents.retriever import hybrid_search
from agents.responder import generate_response

RESUME_PROMPT = """Com base nesse histórico: [{history}], e nesse input: [{question}], crie APENAS um input resumindo o que o usuário deseja saber para ser utilizado em um agente de LLM. NUNCA responda a pergunta do usuário."""


# --- Preparação do ambiente ---

def ensure_knowledge_base():
    if not os.path.exists(CHROMA_PATH) or not os.path.exists(TOPICS_PATH):
        print("Base de conhecimento não encontrada. Construindo agora...")
        print("Isso pode levar alguns minutos na primeira execução.\n")
        from kb_builder import main as build_kb
        build_kb()
        print()


# --- Core: retorna resultado estruturado ---

def run(user_key: str, question: str) -> dict:
    """Processa uma pergunta e retorna resultado estruturado."""
    ensure_knowledge_base()

    conn = init_db()
    model = ChatOllama(model=CHAT_MODEL)
    router_model = ChatOllama(model=CHAT_MODEL, temperature=0)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    with open(TOPICS_PATH, encoding="utf-8") as f:
        topics: list[str] = json.load(f)["topics"]

    # Etapa 1: roteamento na pergunta original
    topic = route_question(question, topics, router_model)

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Etapa 2: enriquece com histórico se a pergunta for em escopo
    history = load_history(conn, user_key)
    if history:
        query = (ChatPromptTemplate.from_template(RESUME_PROMPT) | model).invoke(
            {"history": history, "question": question}
        ).content.strip()
    else:
        query = question

    # Etapa 3: busca híbrida
    display_topic = topic if topic != GLOBAL_SEARCH else "Computação Gráfica"
    results, search_type = hybrid_search(db, query, topic)

    if not results or results[0][1] < NO_INFO_THRESHOLD:
        conn.close()
        return {
            "response": "Não encontrei informações suficientes sobre esse assunto na base de conhecimento.",
            "topic": display_topic,
            "search_type": search_type,
            "sources": [],
            "out_of_scope": False,
        }

    # Etapa 4: geração da resposta
    response = generate_response(query, display_topic, results, model)
    sources = list(set(doc.metadata.get("source", "") for doc, _ in results))

    save_history(conn, user_key, question, response)
    conn.close()

    return {
        "response": response,
        "topic": display_topic,
        "search_type": search_type,
        "sources": [s for s in sources if s],
        "out_of_scope": False,
    }


# --- CLI ---

def _print_result(result: dict):
    if result["out_of_scope"] or not result["sources"]:
        print(result["response"])
        return
    print(f"[Tópico: {result['topic']} | Busca: {result['search_type']}]")
    print(result["response"])
    if result["sources"]:
        print("\nFontes:\n" + "\n".join(result["sources"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatbot de monitoria de Computação Gráfica")
    parser.add_argument("user_key", help="Identificador único do usuário")
    parser.add_argument("question", help="Pergunta a ser respondida")
    args = parser.parse_args()
    _print_result(run(user_key=args.user_key, question=args.question))
