from langchain_chroma import Chroma
from config import HYBRID_THRESHOLD


def hybrid_search(db: Chroma, query: str, topic: str, k: int = 5) -> tuple[list, str]:
    filtered = db.similarity_search_with_relevance_scores(
        query, k=k, filter={"topic": topic}
    )
    if filtered and filtered[0][1] >= HYBRID_THRESHOLD:
        return filtered, "filtrado"
    return db.similarity_search_with_relevance_scores(query, k=k), "global"
