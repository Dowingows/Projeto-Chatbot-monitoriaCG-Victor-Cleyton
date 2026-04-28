import re

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from config import NUM_CANDIDATES, CHAT_MODEL

CANDIDATE_PROMPT = """Você é um professor de Computação Gráfica especializado em {topic}.

REGRAS OBRIGATÓRIAS:
- Baseie sua resposta EXCLUSIVAMENTE nas passagens do contexto abaixo.
- O contexto pode estar em inglês — traduza os trechos relevantes e explique em português.
- NÃO invente informações que não estejam no contexto.
- Se o contexto for insuficiente para algum ponto, omita-o silenciosamente.

FORMATO — escolha conforme o tipo de pergunta:

• Pergunta conceitual ("o que é", "como funciona", "explique"):
  Use as seções em Markdown:
  ## O que é
  ## Aplicações
  ## Exemplo
  ## Analogia

• Pergunta factual ou histórica ("quem criou", "quando surgiu", "qual o nome"):
  Responda em 1 a 3 parágrafos diretos, sem seções forçadas.

• Pergunta procedural ("como fazer", "quais os passos", "como implementar"):
  Use lista numerada com os passos.

• Pergunta comparativa ("qual a diferença", "compare", "vantagens e desvantagens"):
  Use bullet points paralelos ou tabela Markdown.

---
Contexto:
{context}

Pergunta: {question}

Resposta:"""

CRITIC_PROMPT = """Você é um avaliador de respostas educacionais sobre Computação Gráfica.

Pergunta: {question}

Avalie as {n} respostas abaixo e escolha a que:
1. Responde corretamente e diretamente à pergunta
2. Está baseada em conteúdo técnico real de Computação Gráfica (não em conhecimento genérico)
3. É mais precisa e completa

Responda APENAS com o número da melhor resposta (ex: 1 ou 2).

{candidates}

Melhor resposta (apenas o número):"""


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
        "n": n,
        "candidates": numbered,
    }).content.strip()

    match = re.search(r"\d", raw)
    idx = int(match.group()) - 1 if match else 0
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
