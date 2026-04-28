import os
import json
import shutil

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from config import DOCS_PATH, CHROMA_PATH, TOPICS_PATH, EMBED_MODEL, CHAT_MODEL

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


def extract_topic(filename: str, sample: str, model: ChatOllama) -> str:
    # Usa chunks intermediários para evitar capas/metadados
    return (ChatPromptTemplate.from_template(TOPIC_PROMPT) | model).invoke(
        {"filename": filename, "sample": sample[:300]}
    ).content.strip()


def main():
    print("Carregando documentos...")
    loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
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

    # Agrupa chunks por arquivo para classificar 1 vez por arquivo
    file_chunks: dict[str, list] = {}
    for chunk in chunks:
        file_chunks.setdefault(chunk.metadata.get("source", ""), []).append(chunk)

    print(f"\nClassificando {len(file_chunks)} arquivos com LLM...")
    model = ChatOllama(model=CHAT_MODEL, temperature=0)
    file_topics: dict[str, str] = {}
    for source, file_docs in file_chunks.items():
        filename = os.path.basename(source)
        mid = max(1, len(file_docs) // 3)  # skip cover/metadata pages
        topic = extract_topic(filename, file_docs[mid].page_content, model)
        file_topics[source] = topic
        print(f"  {filename} → {topic}")

    for chunk in chunks:
        chunk.metadata["topic"] = file_topics.get(chunk.metadata.get("source", ""), "Geral")

    print("\nCriando base vetorial ChromaDB...")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)

    unique_topics = sorted(set(file_topics.values()))
    os.makedirs(CHROMA_PATH, exist_ok=True)
    with open(TOPICS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "topics": unique_topics,
            "file_topics": {os.path.basename(k): v for k, v in file_topics.items()},
        }, f, ensure_ascii=False, indent=2)

    print(f"\nBase pronta: {len(chunks)} chunks | {len(unique_topics)} tópicos")
    print(f"Tópicos: {unique_topics}")
