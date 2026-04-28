from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config import OUT_OF_SCOPE, GLOBAL_SEARCH

ROUTER_PROMPT = """Você é um classificador de perguntas da disciplina de Computação Gráfica.

Classifique a pergunta em um dos tópicos disponíveis abaixo.
Responda FORA_DE_ESCOPO APENAS se a pergunta for claramente sobre um assunto sem nenhuma relação com Computação Gráfica (ex: história, culinária, biologia, esportes).

Qualquer pergunta que mencione ou envolva conceitos como: pipeline, renderização, geometria, transformações, OpenGL, iluminação, viewing, rasterização, pixels, vértices, matrizes, projeção, shaders, dispositivos gráficos ou qualquer outro tema de CG deve ser classificada em um dos tópicos da lista — mesmo que você não tenha certeza do tópico exato.

Tópicos disponíveis:
{topics}

Pergunta: {question}

Responda APENAS com o nome exato de um tópico da lista ou FORA_DE_ESCOPO:"""

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
    if result in topics:
        return result
    # Resultado não reconhecido mas não é claramente fora de escopo:
    # faz busca global sem filtro de tópico
    return GLOBAL_SEARCH


def identify_area(question: str, model: ChatOllama) -> str:
    return (ChatPromptTemplate.from_template(AREA_PROMPT) | model).invoke(
        {"question": question}
    ).content.strip()
