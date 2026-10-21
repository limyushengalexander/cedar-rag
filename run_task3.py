"""Execute Task 3 retrieval runs without changing either retriever."""

from contextlib import redirect_stdout
import csv
import hashlib
from importlib.metadata import version
import io
from pathlib import Path
import platform
import re
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

import embedding_retriever as embedding
import keyword_retriever as keyword


QUESTIONS = (
    ("Q1", "May a trained graduate student operate the laser alone at 19:00?", "A1, A2"),
    ("Q2", "Can I use the laser without another person present?", "A2"),
    ("Q3", "How frequently must I refresh my laser qualification?", "T1"),
    ("Q4", "What is the laser wavelength?", "absent"),
)
FIELDS = (
    "question_id", "keyword_top_ids", "keyword_scores", "embedding_top_ids",
    "embedding_scores", "required_ids", "keyword_evidence_complete",
    "embedding_evidence_complete", "observation",
)


def capture_keyword(question, k):
    """Call the unchanged keyword CLI and capture its printed output."""
    original_argv = sys.argv
    output = io.StringIO()
    try:
        sys.argv = [str(Path(keyword.__file__)), question, "--k", str(k)]
        with redirect_stdout(output):
            keyword.main()
    finally:
        sys.argv = original_argv
    return output.getvalue()


def capture_embedding(model, chunks, documents, question, k):
    output = io.StringIO()
    with redirect_stdout(output):
        embedding.inspect_question(model, chunks, documents, question, k)
    return output.getvalue()


def selected_pairs(output):
    """Read the actual printed selections and their ranked scores, in order."""
    ranked_scores = dict(re.findall(
        r"^([A-Z]\d+) \| [^\n|]+ \| (?:score|similarity)=([^\s|]+)",
        output, re.MULTILINE,
    ))
    if "Selected passages" not in output:
        if "No matching passages." not in output:
            raise ValueError("Missing selection output")
        return []
    selected_text = output.split("Selected passages", 1)[1]
    ids = re.findall(r"^([A-Z]\d+) \| [^\n|]+$", selected_text, re.MULTILINE)
    return [(chunk_id, ranked_scores[chunk_id]) for chunk_id in ids]


def compact(pairs):
    return "; ".join(f"{chunk_id} ({score})" for chunk_id, score in pairs) or "(none)"


def main():
    root = Path(__file__).resolve().parent
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=True)
    protected = [root / name for name in ("keyword_retriever.py", "embedding_retriever.py")]
    protected += sorted((root / "docs").glob("*.txt"))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}

    chunks = keyword.load_chunks(root / "docs")
    model = SentenceTransformer(
        embedding.MODEL_NAME, device="cpu",
        cache_folder=str(root / ".venv" / "model-cache"), local_files_only=True,
    )
    if str(model.device) != "cpu":
        raise ValueError(f"Expected CPU, got {model.device}")
    # Same encoding and arithmetic as Task 2; called once for the entire run.
    documents = model.encode_document(
        [chunk.text for chunk in chunks], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype(np.float64)
    if documents.shape != (6, 384):
        raise ValueError(f"Unexpected document shape: {documents.shape}")
    embedding.verify_lengths(documents)
    header = (
        f"Model: {embedding.MODEL_NAME}\nDevice: {model.device}\n"
        f"Python: {platform.python_version()}\n"
        + "".join(f"{package}: {version(package)}\n" for package in
                  ("sentence-transformers", "numpy", "torch"))
        + f"Document matrix shape: {documents.shape}\n"
        + "Document embeddings computed once and reused for every question and k run.\n"
        + "Retrieval uses all 384 coordinates; calculations use unrounded vectors.\n"
    )

    rows = []
    comparisons = []
    q1_embedding = None
    with (evidence / "retrieval-runs.txt").open("w", encoding="utf-8", newline="\n") as log:
        log.write(header)
        for question_id, question, required in QUESTIONS:
            keyword_output = capture_keyword(question, 2)
            embedding_output = capture_embedding(model, chunks, documents, question, 2)
            for method, output in (("KEYWORD", keyword_output), ("EMBEDDING", embedding_output)):
                log.write(f"\n{'=' * 20} {question_id} | {method} | k=2 {'=' * 20}\n")
                log.write(output)
            keyword_pairs = selected_pairs(keyword_output)
            embedding_pairs = selected_pairs(embedding_output)
            rows.append(dict(zip(FIELDS, (
                question_id,
                "; ".join(pair[0] for pair in keyword_pairs),
                "; ".join(pair[1] for pair in keyword_pairs),
                "; ".join(pair[0] for pair in embedding_pairs),
                "; ".join(pair[1] for pair in embedding_pairs),
                required, "", "", "",
            ))))
            comparisons.append((question_id, keyword_pairs, embedding_pairs, required))
            if question_id == "Q1":
                q1_embedding = embedding_output

    with (root / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    experiment = []
    with (evidence / "k-experiment.txt").open("w", encoding="utf-8", newline="\n") as log:
        log.write(header)
        for k in (1, 2, 3):
            output = q1_embedding if k == 2 else capture_embedding(
                model, chunks, documents, QUESTIONS[0][1], k,
            )
            pairs = selected_pairs(output)
            log.write(f"\n{'=' * 20} Q1 | EMBEDDING | k={k} {'=' * 20}\n")
            log.write(output)
            log.write("\nSelected IDs and corresponding scores (rank order):\n")
            for chunk_id, score in pairs:
                log.write(f"{chunk_id} | similarity={score}\n")
            experiment.append((k, pairs))

    for path, digest in before.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Protected file changed: {path}")

    print("Question | Keyword selections (score) | Embedding selections (score) | Required IDs")
    for question_id, keyword_pairs, embedding_pairs, required in comparisons:
        print(f"{question_id} | {compact(keyword_pairs)} | {compact(embedding_pairs)} | {required}")
    print("\nQ1 embedding k experiment | Selected IDs (score)")
    for k, pairs in experiment:
        print(f"k={k} | {compact(pairs)}")
    print("\nSaved evidence/retrieval-runs.txt, results.csv, and evidence/k-experiment.txt.")
    print("Verified: source documents and both retrieval methods unchanged.")
    print("Judgment fields and interpretation left blank. Q4 required evidence: absent.")


if __name__ == "__main__":
    main()
