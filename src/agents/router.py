from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config import OUT_OF_SCOPE

ROUTER_PROMPT = """Você é um classificador de perguntas sobre Computação Gráfica.

Se a pergunta estiver relacionada a Computação Gráfica, classifique-a no tópico mais adequado da lista.
Se a pergunta NÃO for sobre Computação Gráfica, responda exatamente: FORA_DE_ESCOPO

Tópicos disponíveis:
{topics}

Pergunta: {question}

Responda APENAS com o nome exato do tópico da lista ou FORA_DE_ESCOPO, sem nenhuma explicação:"""

AREA_PROMPT = """Identifique a área do conhecimento da seguinte pergunta em 2-4 palavras em português.
Responda APENAS com o nome da área, sem explicações.

Pergunta: {question}

Área:"""


def route_question(question: str, topics: list[str], model: ChatOllama) -> str:
    result = (ChatPromptTemplate.from_template(ROUTER_PROMPT) | model).invoke({
        "topics": "\n".join(f"- {t}" for t in topics),
        "question": question,
    }).content.strip()

    if result == OUT_OF_SCOPE:
        return OUT_OF_SCOPE
    return result if result in topics else OUT_OF_SCOPE


def identify_area(question: str, model: ChatOllama) -> str:
    return (ChatPromptTemplate.from_template(AREA_PROMPT) | model).invoke(
        {"question": question}
    ).content.strip()
