"""Task 2: inspect real CPU embeddings and retrieve passages by cosine similarity."""

import argparse
from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np
from sentence_transformers import SentenceTransformer

from keyword_retriever import load_chunks


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NORM_TOLERANCE = 1e-6


def coordinates(vector):
    return "[" + ", ".join(f"{value:.4f}" for value in vector[:12]) + "]"


def verify_lengths(vectors):
    lengths = np.linalg.norm(vectors, axis=-1)
    if not np.allclose(lengths, 1.0, rtol=0, atol=NORM_TOLERANCE):
        raise ValueError(f"Vector lengths are not approximately 1: {lengths}")
    return lengths


def inspect_question(model, chunks, documents, question, k):
    # Preserve every stored embedding coordinate; use float64 for arithmetic.
    query = model.encode_query(
        question, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float64)
    if query.shape != (384,):
        raise ValueError(f"Unexpected question vector shape: {query.shape}")
    query_length = float(verify_lengths(query))
    scores = documents @ query
    ranked = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i].chunk_id))

    print(f"\nQuestion: {question}")
    print(f"k: {k}")
    print(f"Question vector shape: {query.shape}")
    print(f"Question first 12 coordinates: {coordinates(query)}")
    print(f"Question vector length: {query_length:.12f}")
    print(f"Question length approximately 1: PASS (absolute tolerance {NORM_TOLERANCE:g})")
    print("\nAll six similarities (descending score, then ascending chunk ID):")
    for i in ranked:
        chunk = chunks[i]
        print(f"{chunk.chunk_id} | {chunk.filename} | similarity={scores[i]:.12f}")

    best = ranked[0]
    document = documents[best]
    products = query * document
    first_subtotal = float(np.sum(products[:12]))
    remaining_subtotal = float(np.sum(products[12:]))
    full_score = float(scores[best])
    document_length = float(np.linalg.norm(document))
    cosine = float(np.dot(query, document) / (query_length * document_length))
    agreement = bool(np.isclose(cosine, full_score, rtol=0, atol=NORM_TOLERANCE))
    if not agreement:
        raise ValueError(f"Cosine {cosine} and dot product {full_score} disagree")
    if not np.isclose(first_subtotal + remaining_subtotal, full_score, rtol=0, atol=1e-12):
        raise ValueError("Coordinate subtotals disagree with the full dot product")

    print(f"\nCoordinate products for top passage {chunks[best].chunk_id}:")
    print("Coordinates are numbered from 1; no semantic labels are assigned.")
    for j in range(12):
        print(f"{j + 1:02d}: {query[j]:.4f} * {document[j]:.4f} = {products[j]:.12f}")
    print("Products and totals use unrounded coordinates; displayed operands are rounded.")
    print(f"First 12 products subtotal: {first_subtotal:.12f}")
    print(f"Remaining 372 products subtotal: {remaining_subtotal:.12f}")
    print(f"Full similarity (normalized dot product): {full_score:.12f}")
    print(f"Direct cosine = dot / (query length * document length): {cosine:.12f}")
    print(f"Lengths used: query={query_length:.12f}, document={document_length:.12f}")
    print(f"Absolute cosine/dot difference: {abs(cosine - full_score):.12g}")
    print(f"Cosine agrees with dot product: PASS (absolute tolerance {NORM_TOLERANCE:g})")

    print("\nSelected passages (no relevance threshold):")
    for i in ranked[:k]:
        chunk = chunks[i]
        print(f"\n{chunk.chunk_id} | {chunk.filename}")
        print(chunk.text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", nargs="+", help="One or more individually quoted questions.")
    parser.add_argument("--k", type=int, default=2, help="Top passages per question (default: 2).")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be a positive integer")

    root = Path(__file__).resolve().parent
    chunks = load_chunks(root / "docs")
    model = SentenceTransformer(
        MODEL_NAME, device="cpu", cache_folder=str(root / ".venv" / "model-cache"),
        local_files_only=True,
    )
    if str(model.device) != "cpu":
        raise ValueError(f"Expected CPU execution, got {model.device}")
    # Encode only passage text, once, before processing any questions.
    documents = model.encode_document(
        [chunk.text for chunk in chunks], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype(np.float64)
    if documents.shape != (6, 384):
        raise ValueError(f"Unexpected document matrix shape: {documents.shape}")
    lengths = verify_lengths(documents)

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {model.device}")
    print(f"Python: {platform.python_version()}")
    for package in ("sentence-transformers", "numpy", "torch"):
        print(f"{package}: {version(package)}")
    print(f"Document matrix shape: {documents.shape}")
    print("Retrieval uses all 384 coordinates, not just the first 12 displayed.")
    print("Embedding coordinates retain model precision; calculations use float64; rounding is display-only.")
    print("\nDocument vectors:")
    for chunk, vector, length in zip(chunks, documents, lengths):
        print(f"{chunk.chunk_id} | {chunk.filename} | first 12 coordinates: {coordinates(vector)}")
        print(f"  Vector length: {length:.12f}")
    print(f"All document lengths approximately 1: PASS (absolute tolerance {NORM_TOLERANCE:g})")
    for question in args.questions:
        inspect_question(model, chunks, documents, question, args.k)


if __name__ == "__main__":
    main()
