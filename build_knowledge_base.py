import argparse
import json
import logging
import sys
from pathlib import Path

from config import (
    KNOWLEDGE_COLLECTION_NAME,
    KNOWLEDGE_EMBEDDING_MODEL,
    KNOWLEDGE_PERSIST_DIR,
)
from rag_knowledge_base import init_knowledge_base


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build or load the CHD clinical guidance knowledge base."
    )
    parser.add_argument("pdf_directory", help="Directory containing CHD-related PDF files.")
    parser.add_argument(
        "--persist-dir",
        default=KNOWLEDGE_PERSIST_DIR,
        help=f"Chroma persistence directory. Default: {KNOWLEDGE_PERSIST_DIR}",
    )
    parser.add_argument(
        "--embedding-model",
        default=KNOWLEDGE_EMBEDDING_MODEL,
        help=f"Local sentence-transformers model. Default: {KNOWLEDGE_EMBEDDING_MODEL}",
    )
    parser.add_argument(
        "--collection-name",
        default=KNOWLEDGE_COLLECTION_NAME,
        help=f"Chroma collection name. Default: {KNOWLEDGE_COLLECTION_NAME}",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete the existing local vector store and rebuild it from PDF files.",
    )
    args = parser.parse_args()

    try:
        logger.info("Initializing CHD knowledge base from %s", args.pdf_directory)
        stats = init_knowledge_base(
            pdf_directory=args.pdf_directory,
            persist_dir=args.persist_dir,
            embedding_model_name=args.embedding_model,
            collection_name=args.collection_name,
            force_rebuild=args.force_rebuild,
        )
    except Exception as exc:
        logger.exception("Knowledge base build failed")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    pdf_count = len(list(Path(args.pdf_directory).glob("*.pdf")))
    print("Knowledge base ready")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(
        f"Processed {pdf_count} PDF files and stored {stats['vector_count']} vectors "
        f"in {stats['persist_dir']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
