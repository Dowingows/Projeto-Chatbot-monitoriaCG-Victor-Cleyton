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
NO_INFO_THRESHOLD = 0.1

GLOBAL_SEARCH = "GLOBAL"

# Número de candidatos gerados pelo responder (critic escolhe o melhor)
NUM_CANDIDATES = 2

# Subpastas de DOCS_PATH a incluir na KB (None = todas)
# Ex: export DOCS_INCLUDE=Livros  →  carrega só Docs/Livros/
DOCS_INCLUDE = [p.strip() for p in os.getenv("DOCS_INCLUDE", "").split(",") if p.strip()] or None

# Páginas por grupo para classificação de tópico em arquivos grandes
PAGE_GROUP_SIZE = int(os.getenv("PAGE_GROUP_SIZE", "15"))

# Arquivo com mais páginas únicas que isso usa classificação por seção
LARGE_FILE_THRESHOLD = int(os.getenv("LARGE_FILE_THRESHOLD", "50"))
