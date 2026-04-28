import os
import json
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DOCS_PATH = "Docs"
CHROMA_PATH = "chroma"
TOPICS_PATH = "chroma/topics.json"
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")

TOPIC_PROMPT = """Você é um especialista em Computação Gráfica.
Analise o nome e a amostra de conteúdo deste arquivo e responda com um rótulo genérico e curto (2-4 palavras em português) que descreva o tópico principal.
Responda APENAS com o rótulo, sem explicações ou pontuação extra.

Nome do arquivo: {filename}
Amostra do conteúdo: {sample}

Tópico:"""


def extract_topic(filename: str, sample: str, model: ChatOllama) -> str:
    prompt = ChatPromptTemplate.from_template(TOPIC_PROMPT)
    response = (prompt | model).invoke({"filename": filename, "sample": sample[:500]})
    return response.content.strip()


def main():
    loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    # Agrupa chunks por arquivo fonte
    file_chunks: dict[str, list] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        file_chunks.setdefault(source, []).append(chunk)

    # Classifica cada arquivo com 1 chamada ao LLM
    model = ChatOllama(model=CHAT_MODEL, temperature=0)
    file_topics: dict[str, str] = {}

    print(f"Classificando {len(file_chunks)} arquivos...")
    for source, file_docs in file_chunks.items():
        filename = os.path.basename(source)
        sample = file_docs[0].page_content
        topic = extract_topic(filename, sample, model)
        file_topics[source] = topic
        print(f"  {filename} → {topic}")

    # Aplica o tópico como metadata em cada chunk
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        chunk.metadata["topic"] = file_topics.get(source, "Geral")

    # Recria o banco ChromaDB
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)

    # Salva lista de tópicos para o router do chatbot
    unique_topics = sorted(set(file_topics.values()))
    topics_data = {
        "topics": unique_topics,
        "file_topics": {os.path.basename(k): v for k, v in file_topics.items()},
    }
    with open(TOPICS_PATH, "w", encoding="utf-8") as f:
        json.dump(topics_data, f, ensure_ascii=False, indent=2)

    print(f"\nBase criada: {len(chunks)} chunks, {len(unique_topics)} tópicos")
    print(f"Tópicos descobertos: {unique_topics}")


if __name__ == "__main__":
    main()
