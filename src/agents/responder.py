from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

ANSWER_PROMPT = """Você é um professor de Computação Gráfica explicando o tópico "{topic}" para um aluno de graduação.
Responda com base APENAS no contexto fornecido, sem inventar informações.

Estruture sua resposta em Markdown com as seguintes seções, quando aplicável:

## O que é
Explique o conceito de forma clara e direta.

## Aplicações
Liste as principais aplicações práticas (use bullet points).

## Exemplo
Dê um exemplo concreto do conceito em ação.

## Analogia
Faça uma analogia com algo do mundo real para facilitar a compreensão.

---

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
