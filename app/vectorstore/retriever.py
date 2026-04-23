from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import ChromaStore

class Retriever:
    def __init__(self, store: ChromaStore, embedder: Embedder, top_k: int = 5):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, question: str) -> list[dict]:
        q_emb = self.embedder.embed_query(question)
        results = self.store.query(q_emb, top_k=self.top_k)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        items: list[dict] = []
        for i in range(len(documents)):
            items.append(
                {
                    "text": documents[i],
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": distances[i] if i < len(distances) else None,
                }
            )

        return items