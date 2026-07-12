from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "chd_clinical_guidance"

_VECTOR_STORE = None
_KB_STATUS = {
    "ready": False,
    "status": "not_initialized",
    "pdf_count": 0,
    "chunk_count": 0,
    "vector_count": 0,
    "persist_dir": None,
    "collection_name": DEFAULT_COLLECTION_NAME,
}


def _build_embeddings(model_name: str):
    from langchain_core.embeddings import Embeddings
    from sentence_transformers import SentenceTransformer

    class LocalSentenceTransformerEmbeddings(Embeddings):
        def __init__(self, name: str):
            self.model = SentenceTransformer(name)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            vectors = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vectors.tolist()

        def embed_query(self, text: str) -> list[float]:
            vector = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vector.tolist()

    return LocalSentenceTransformerEmbeddings(model_name)


def _load_pdf_documents(pdf_paths: list[Path]):
    from langchain_core.documents import Document
    from pypdf import PdfReader

    documents = []
    processed_paths = []
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(str(pdf_path))
            processed_paths.append(pdf_path)
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "path": str(pdf_path),
                            "page": page_number,
                        },
                    )
                )
        except Exception:
            logger.exception("Failed to parse PDF: %s", pdf_path)
    return documents, processed_paths


def _split_documents(documents, chunk_size: int, chunk_overlap: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def _build_vector_store(
    persist_dir: Path, collection_name: str, embedding_model_name: str
):
    from langchain_chroma import Chroma

    embeddings = _build_embeddings(embedding_model_name)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def _vector_count(vector_store) -> int:
    try:
        return int(vector_store._collection.count())
    except Exception:
        logger.exception("Failed to read vector count from Chroma.")
        return 0


def init_knowledge_base(
    pdf_directory,
    persist_dir,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = 800,
    chunk_overlap: int = 80,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    global _VECTOR_STORE, _KB_STATUS

    pdf_dir = Path(pdf_directory).resolve()
    persist_path = Path(persist_dir).resolve()

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files were found in: {pdf_dir}")

    if force_rebuild and persist_path.exists():
        try:
            shutil.rmtree(persist_path)
        except PermissionError as exc:
            raise RuntimeError(
                "Cannot rebuild the knowledge base because Chroma is in use. "
                "Stop the Flask service or any other process using chroma_db, then retry."
            ) from exc

    vector_store = _build_vector_store(
        persist_path, collection_name, embedding_model_name
    )
    existing_count = _vector_count(vector_store)
    if existing_count > 0 and not force_rebuild:
        _VECTOR_STORE = vector_store
        _KB_STATUS = {
            "ready": True,
            "status": "loaded",
            "pdf_count": len(pdf_paths),
            "chunk_count": existing_count,
            "vector_count": existing_count,
            "persist_dir": str(persist_path),
            "collection_name": collection_name,
        }
        logger.info(
            "Loaded existing knowledge base from %s with %s vectors.",
            persist_path,
            existing_count,
        )
        return dict(_KB_STATUS)

    documents, processed_paths = _load_pdf_documents(pdf_paths)
    if not documents:
        raise RuntimeError(
            f"No extractable PDF text was found in: {pdf_dir}. Check the PDF files."
        )

    chunks = _split_documents(documents, chunk_size, chunk_overlap)
    vector_store.add_documents(chunks)
    final_count = _vector_count(vector_store)

    _VECTOR_STORE = vector_store
    _KB_STATUS = {
        "ready": True,
        "status": "built",
        "pdf_count": len(processed_paths),
        "chunk_count": len(chunks),
        "vector_count": final_count,
        "persist_dir": str(persist_path),
        "collection_name": collection_name,
    }
    logger.info(
        "Built knowledge base from %s PDFs into %s with %s chunks.",
        len(processed_paths),
        persist_path,
        len(chunks),
    )
    return dict(_KB_STATUS)


def query_knowledge_base(query: str, top_k: int = 3) -> list[dict[str, str]]:
    if _VECTOR_STORE is None:
        raise RuntimeError("Knowledge base has not been initialized.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    matches = _VECTOR_STORE.similarity_search(query, k=top_k)
    formatted = []
    for doc in matches:
        content = " ".join(doc.page_content.split())
        formatted.append(
            {
                "content": content,
                "source": str(doc.metadata.get("source", "unknown")),
            }
        )
    return formatted


def get_knowledge_base_status() -> dict[str, Any]:
    return dict(_KB_STATUS)
