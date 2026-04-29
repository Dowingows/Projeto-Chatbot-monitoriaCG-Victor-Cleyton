import re

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from config import NUM_CANDIDATES, CHAT_MODEL

CANDIDATE_PROMPT = """Você é uma professora de Computação Gráfica explicando sobre {topic} para um aluno de graduação.

REGRAS:
- Use EXCLUSIVAMENTE as informações do contexto abaixo. Não invente.
- O contexto pode estar em inglês — leia, entenda e explique em português.
- Responda de forma natural e fluida, como numa conversa de monitoria.
- Não use seções fixas, títulos ou templates. Escreva em prosa ou com listas quando fizer sentido organicamente.
- Se o contexto for insuficiente para responder, diga o que sabe pelo contexto e indique a limitação.

Contexto:
{context}

Pergunta: {question}

Resposta:"""

CRITIC_PROMPT = """Você é uma professora avaliando respostas de monitoria sobre Computação Gráfica.

Pergunta do aluno: {question}

Avalie cada resposta abaixo com uma nota de 0 a 10 considerando:
- Coerência: a resposta faz sentido e é internamente consistente?
- Correção técnica: as informações estão corretas para Computação Gráfica?
- Relevância: responde diretamente o que foi perguntado?
- Fundamentação: está baseada em conteúdo técnico real (não inventado)?

{candidates}

Para cada resposta, dê uma nota. Depois indique qual é a melhor.
Formato obrigatório:
Resposta 1: <nota>/10
Resposta 2: <nota>/10
Melhor: <número>"""


def _split_results(results: list, n: int) -> list[list]:
    """Divide results into n groups; group i gets results[i::n] (interleaved by score rank)."""
    return [results[i::n] for i in range(n) if results[i::n]]


def generate_response(query: str, topic: str, results: list, model: ChatOllama) -> str:
    n = min(NUM_CANDIDATES, len(results))
    groups = _split_results(results, n)

    candidates = []
    for group in groups:
        context = "\n\n---\n\n".join(doc.page_content for doc, _ in group)
        response = (ChatPromptTemplate.from_template(CANDIDATE_PROMPT) | model).invoke(
            {"topic": topic, "context": context, "question": query}
        ).content
        candidates.append(response)

    if len(candidates) == 1:
        return candidates[0]

    numbered = "\n\n".join(f"=== Resposta {i + 1} ===\n{c}" for i, c in enumerate(candidates))
    critic = ChatOllama(model=CHAT_MODEL, temperature=0)
    raw = (ChatPromptTemplate.from_template(CRITIC_PROMPT) | critic).invoke({
        "question": query,
        "candidates": numbered,
    }).content.strip()

    # Tenta ler "Melhor: N", senão cai para o primeiro dígito encontrado
    best_match = re.search(r"(?i)melhor\s*[:=]\s*(\d+)", raw)
    if best_match:
        idx = int(best_match.group(1)) - 1
    else:
        fallback = re.search(r"\d", raw)
        idx = int(fallback.group()) - 1 if fallback else 0
    idx = max(0, min(idx, len(candidates) - 1))
    return candidates[idx]


def out_of_scope_message(area: str, topics: list[str]) -> str:
    sample = ", ".join(topics[:5])
    if len(topics) > 5:
        sample += ", entre outros"
    return (
        f"Sua pergunta parece ser sobre {area}. "
        f"Este chatbot é especializado em Computação Gráfica e responde apenas dúvidas relacionadas a essa disciplina. "
        f"Posso ajudar com tópicos como: {sample}."
    )
