import os
import json
import argparse

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from config import (
    CHROMA_PATH, TOPICS_PATH, CHAT_MODEL, EMBED_MODEL,
    NO_INFO_THRESHOLD, OUT_OF_SCOPE,
)
from agents.history import init_db, load_history, save_history
from agents.router import route_question, identify_area
from agents.retriever import hybrid_search
from agents.responder import generate_response, out_of_scope_message

RESUME_PROMPT = """Com base nesse histórico: [{history}], e nesse input: [{question}], crie APENAS um input resumindo o que o usuário deseja saber para ser utilizado em um agente de LLM. NUNCA responda a pergunta do usuário."""


# --- Preparação do ambiente ---

def ensure_knowledge_base():
    if not os.path.exists(CHROMA_PATH) or not os.path.exists(TOPICS_PATH):
        print("Base de conhecimento não encontrada. Construindo agora...")
        print("Isso pode levar alguns minutos na primeira execução.\n")
        from kb_builder import main as build_kb
        build_kb()
        print()


# --- Orquestração ---

def main(user_key: str, question: str):
    # Etapa 1: garante que a base existe
    ensure_knowledge_base()

    # Etapa 2: inicializa recursos
    conn = init_db()
    model = ChatOllama(model=CHAT_MODEL)
    router_model = ChatOllama(model=CHAT_MODEL, temperature=0)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    with open(TOPICS_PATH, encoding="utf-8") as f:
        topics: list[str] = json.load(f)["topics"]

    # Etapa 3: roteamento na pergunta original (antes de qualquer processamento)
    topic = route_question(question, topics, router_model)

    if topic == OUT_OF_SCOPE:
        area = identify_area(question, router_model)
        print(out_of_scope_message(area, topics))
        conn.close()
        return

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Etapa 4: enriquece com histórico apenas se a pergunta for em escopo
    history = load_history(conn, user_key)
    if history:
        query = (ChatPromptTemplate.from_template(RESUME_PROMPT) | model).invoke(
            {"history": history, "question": question}
        ).content.strip()
    else:
        query = question

    # Etapa 5: busca híbrida na base de conhecimento
    results, search_type = hybrid_search(db, query, topic)

    if not results or results[0][1] < NO_INFO_THRESHOLD:
        print("Não encontrei informações suficientes sobre esse assunto na base de conhecimento.")
        conn.close()
        return

    # Etapa 6: geração da resposta
    response = generate_response(query, topic, results, model)
    sources = list(set(doc.metadata.get("source", "") for doc, _ in results))

    # Etapa 7: salva histórico e exibe resultado
    save_history(conn, user_key, question, response)
    conn.close()

    print(f"[Tópico: {topic} | Busca: {search_type}]")
    print(response)
    print("\nFontes:\n" + "\n".join(sources))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatbot de monitoria de Computação Gráfica")
    parser.add_argument("user_key", help="Identificador único do usuário")
    parser.add_argument("question", help="Pergunta a ser respondida")
    args = parser.parse_args()
    main(user_key=args.user_key, question=args.question)
