import os
import re
import json
import shutil

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from config import (
    DOCS_PATH, CHROMA_PATH, TOPICS_PATH, EMBED_MODEL, CHAT_MODEL,
    DOCS_INCLUDE, PAGE_GROUP_SIZE, LARGE_FILE_THRESHOLD,
)

# ── Prompts ───────────────────────────────────────────────────────────────────

TOPIC_PROMPT = """Você é um especialista em Computação Gráfica.
Analise o NOME DO ARQUIVO abaixo e defina o tópico técnico de Computação Gráfica que ele cobre.

Regras:
- O nome do arquivo é a fonte mais confiável. Priorize-o.
- Responda com 2-4 palavras descrevendo o CONTEÚDO TÉCNICO (ex: "Viewing 2D", "OpenGL", "Transformações 3D", "Iluminação", "Algoritmos de Seleção").
- NUNCA use: Documentação, Material, Guia, Apostila, Slides, PDF, Arquivo, Livro.
- Se o nome não for claro, use a amostra de conteúdo como apoio.
- Responda APENAS com o nome do tópico, sem explicações.

Nome do arquivo: {filename}
Amostra do conteúdo (pode conter metadados, priorize o nome do arquivo): {sample}

Tópico:"""

SECTION_TOPIC_PROMPT = """Você é um especialista em Computação Gráfica.
O trecho abaixo é de um livro técnico (pode estar em inglês).
Classifique o tópico de Computação Gráfica coberto neste trecho.

Tópicos conhecidos da disciplina:
{known_topics}

Regras:
- Escolha o tópico mais próximo da lista acima e use o nome EXATO.
- Se nenhum se encaixar, crie um nome curto (2-4 palavras) em português.
- NUNCA use: Livro, Capítulo, Documentação, Introdução genérica.
- Responda APENAS com o nome do tópico, sem explicações.

Trecho:
{sample}

Tópico:"""

# Tópicos padrão da disciplina, usados quando Docs/Slides/ não existe
_FALLBACK_TOPICS = [
    "Introdução à Computação Gráfica",
    "Histórico da Computação Gráfica",
    "Dispositivos de Entrada",
    "Dispositivos de Saída",
    "OpenGL",
    "Transformações 2D",
    "Algoritmos de Seleção",
    "Viewing 2D",
    "Transformações 3D",
    "Viewing 3D",
    "Modelos de Iluminação",
    "Superfícies Visíveis",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def filter_documents(docs: list, include_folders: list[str]) -> list:
    """Keeps only docs whose source path contains one of the include_folders."""
    filtered = [
        d for d in docs
        if any(
            f"{os.sep}{folder}{os.sep}" in d.metadata.get("source", "") or
            d.metadata.get("source", "").replace("/", os.sep).endswith(f"{os.sep}{folder}")
            for folder in include_folders
        )
    ]
    return filtered


def derive_course_topics(docs_path: str) -> list[str]:
    """Reads slide filenames to build a Portuguese topic list automatically."""
    slides_dir = os.path.join(docs_path, "Slides")
    if not os.path.exists(slides_dir):
        return _FALLBACK_TOPICS
    topics = []
    for fname in sorted(os.listdir(slides_dir)):
        if fname.lower().endswith(".pdf"):
            name = re.sub(r'^\d+\s*[-–]\s*', '', fname)  # remove "01 - "
            name = re.sub(r'\.pdf$', '', name, flags=re.IGNORECASE).strip()
            if name:
                topics.append(name)
    return topics if topics else _FALLBACK_TOPICS


def extract_topic(filename: str, sample: str, model: ChatOllama) -> str:
    return (ChatPromptTemplate.from_template(TOPIC_PROMPT) | model).invoke(
        {"filename": filename, "sample": sample[:300]}
    ).content.strip()


def classify_section(sample: str, known_topics: list[str], model: ChatOllama) -> str:
    topics_str = "\n".join(f"- {t}" for t in known_topics)
    return (ChatPromptTemplate.from_template(SECTION_TOPIC_PROMPT) | model).invoke(
        {"known_topics": topics_str, "sample": sample[:400]}
    ).content.strip()


def build_page_groups(chunks: list) -> dict[int, list]:
    """Groups chunks by page range buckets of PAGE_GROUP_SIZE pages."""
    groups: dict[int, list] = {}
    for chunk in chunks:
        page = chunk.metadata.get("page", 0)
        bucket = (page // PAGE_GROUP_SIZE) * PAGE_GROUP_SIZE
        groups.setdefault(bucket, []).append(chunk)
    return groups


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Carregando documentos...")
    loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if DOCS_INCLUDE:
        documents = filter_documents(documents, DOCS_INCLUDE)
        print(f"  Filtro ativo: {DOCS_INCLUDE}")

    print(f"  {len(documents)} páginas carregadas.")

    print("Dividindo em chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"  {len(chunks)} chunks gerados.")

    known_topics = derive_course_topics(DOCS_PATH)
    model = ChatOllama(model=CHAT_MODEL, temperature=0)

    # Agrupa chunks por arquivo
    file_chunks: dict[str, list] = {}
    for chunk in chunks:
        file_chunks.setdefault(chunk.metadata.get("source", ""), []).append(chunk)

    # Mapeamento chunk → tópico (por chunk para arquivos grandes, por arquivo para pequenos)
    chunk_topics: dict[int, str] = {}  # id(chunk) → topic

    print(f"\nClassificando {len(file_chunks)} arquivo(s) com LLM...")
    file_summary: dict[str, str] = {}

    for source, file_docs in file_chunks.items():
        filename = os.path.basename(source)
        unique_pages = {c.metadata.get("page", 0) for c in file_docs}

        if len(unique_pages) > LARGE_FILE_THRESHOLD:
            # Classificação por seção de PAGE_GROUP_SIZE páginas
            print(f"  {filename} ({len(unique_pages)} págs) → classificação por seção...")
            page_groups = build_page_groups(file_docs)
            for bucket, group_chunks in sorted(page_groups.items()):
                sample = group_chunks[len(group_chunks) // 2].page_content
                topic = classify_section(sample, known_topics, model)
                for chunk in group_chunks:
                    chunk_topics[id(chunk)] = topic
                print(f"    págs {bucket}-{bucket + PAGE_GROUP_SIZE - 1} → {topic}")
            # Para o resumo do arquivo, usa o tópico mais frequente
            topics_in_file = [chunk_topics[id(c)] for c in file_docs]
            file_summary[os.path.basename(source)] = max(set(topics_in_file), key=topics_in_file.count)
        else:
            # Classificação por nome de arquivo (comportamento original)
            mid = max(1, len(file_docs) // 3)
            topic = extract_topic(filename, file_docs[mid].page_content, model)
            for chunk in file_docs:
                chunk_topics[id(chunk)] = topic
            file_summary[os.path.basename(source)] = topic
            print(f"  {filename} → {topic}")

    # Aplica tópico nos metadados de cada chunk
    for chunk in chunks:
        chunk.metadata["topic"] = chunk_topics.get(id(chunk), "Geral")

    print("\nCriando base vetorial ChromaDB...")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)

    unique_topics = sorted({chunk.metadata["topic"] for chunk in chunks})
    os.makedirs(CHROMA_PATH, exist_ok=True)
    with open(TOPICS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "topics": unique_topics,
            "file_topics": file_summary,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nBase pronta: {len(chunks)} chunks | {len(unique_topics)} tópicos")
    print(f"Tópicos: {unique_topics}")
