import os
from dotenv import load_dotenv

load_dotenv()

DOCS_PATH = "Docs"
CHROMA_PATH = "chroma"
TOPICS_PATH = "chroma/topics.json"
DB_PATH = os.getenv("HISTORY_DB", "history.db")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Busca híbrida: score mínimo para preferir resultado filtrado por tópico
HYBRID_THRESHOLD = 0.5
# Score mínimo para considerar que há informação relevante
NO_INFO_THRESHOLD = 0.3

OUT_OF_SCOPE = "FORA_DE_ESCOPO"
