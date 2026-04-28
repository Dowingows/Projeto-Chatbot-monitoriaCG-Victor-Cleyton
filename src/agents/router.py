from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config import GLOBAL_SEARCH

ROUTER_PROMPT = """Você é um classificador de perguntas da disciplina de Computação Gráfica.

Classifique a pergunta no tópico mais relevante da lista abaixo.
Se não encontrar um tópico adequado, escolha o mais próximo ou o mais geral.
Responda APENAS com o nome exato de um tópico da lista, sem nenhuma explicação.

Tópicos disponíveis:
{topics}

Pergunta: {question}

Tópico:"""


def route_question(question: str, topics: list[str], model: ChatOllama) -> str:
    result = (ChatPromptTemplate.from_template(ROUTER_PROMPT) | model).invoke({
        "topics": "\n".join(f"- {t}" for t in topics),
        "question": question,
    }).content.strip()

    return result if result in topics else GLOBAL_SEARCH
