import argparse
from pathlib import Path

from app.utils.config import (
    VECTOR_DB_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from app.utils.logger import get_logger
from app.ingestion.pdf_loader import load_pdf_text
from app.ingestion.text_cleaner import clean_text
from app.ingestion.chunker import chunk_text
from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import ChromaStore
from app.vectorstore.retriever import Retriever

logger = get_logger(__name__)

def build_index(pdf_path: str) -> int:
    pdf_path_obj = Path(pdf_path)

    logger.info("Loading PDF: %s", pdf_path_obj)
    raw_text = load_pdf_text(pdf_path_obj)

    logger.info("Cleaning text")
    cleaned_text = clean_text(raw_text)

    logger.info("Chunking text")
    chunks = chunk_text(
        cleaned_text,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    logger.info("Loading embedder: %s", EMBEDDING_MODEL_NAME)
    embedder = Embedder(EMBEDDING_MODEL_NAME)

    logger.info("Embedding chunks")
    embeddings = embedder.embed_documents(chunks)

    logger.info("Saving into Chroma")
    store = ChromaStore(VECTOR_DB_DIR, CHROMA_COLLECTION_NAME)
    store.upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
        source_name=pdf_path_obj.stem,
    )

    logger.info("Indexed %d chunks", len(chunks))
    return len(chunks)

def search(question: str) -> list[dict]:
    embedder = Embedder(EMBEDDING_MODEL_NAME)
    store = ChromaStore(VECTOR_DB_DIR, CHROMA_COLLECTION_NAME)
    retriever = Retriever(store, embedder, top_k=TOP_K)
    return retriever.retrieve(question)

def main():
    parser = argparse.ArgumentParser(description="Toxic Text Analyzer - RAG backend")
    parser.add_argument("--pdf", type=str, help="Path to PDF to index")
    parser.add_argument("--ask", type=str, help="Question to search against the index")

    args = parser.parse_args()

    if args.pdf:
        count = build_index(args.pdf)
        print(f"Indexed {count} chunks from {args.pdf}")

    if args.ask:
        results = search(args.ask)
        print("\nTop matches:\n")
        for i, item in enumerate(results, start=1):
            print(f"--- Match {i} ---")
            print(f"Distance: {item['distance']}")
            print(f"Source: {item['metadata']}")
            print(item["text"])
            print()

    if not args.pdf and not args.ask:
        parser.print_help()

if __name__ == "__main__":
    main()