from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

ANSWER_PROMPT = """Você é uma IA monitora de Computação Gráfica especializada em {topic}.
Responda a pergunta com base no contexto fornecido.
NÃO utilize informações de fora do contexto. A resposta será impressa em terminal, NÃO use formatação especial.

Contexto:
{context}

Pergunta: {question}

Resposta:"""


def generate_response(query: str, topic: str, results: list, model: ChatOllama) -> str:
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in results)
    return (ChatPromptTemplate.from_template(ANSWER_PROMPT) | model).invoke(
        {"topic": topic, "context": context, "question": query}
    ).content


def out_of_scope_message(area: str, topics: list[str]) -> str:
    sample = ", ".join(topics[:5])
    if len(topics) > 5:
        sample += ", entre outros"
    return (
        f"Sua pergunta parece ser sobre {area}. "
        f"Este chatbot é especializado em Computação Gráfica e responde apenas dúvidas relacionadas a essa disciplina. "
        f"Posso ajudar com tópicos como: {sample}."
    )
