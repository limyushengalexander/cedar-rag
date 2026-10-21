"""Task 1: retrieve labeled passages using distinct keyword overlap only."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re


STOPWORDS = frozenset(
    "a an the is are to of in at for and or may can i it be".split()
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    filename: str
    text: str


def words(text: str) -> set[str]:
    """Lowercase, extract alphanumeric words, and remove only STOPWORDS."""
    return set(re.findall(r"[^\W_]+", text.lower())) - STOPWORDS


def load_chunks(docs_dir: Path) -> list[Chunk]:
    chunks = []
    for filename in ("access.txt", "training.txt", "booking.txt"):
        contents = (docs_dir / filename).read_text(encoding="utf-8")
        for paragraph in re.split(r"\n\s*\n", contents.strip()):
            match = re.fullmatch(r"\[([A-Z]\d+)\] (.+)", paragraph, re.DOTALL)
            if match is None:
                raise ValueError(f"Invalid labeled paragraph in {filename}: {paragraph!r}")
            chunk_id, text = match.groups()
            chunks.append(Chunk(chunk_id, filename, text))
    if len(chunks) != 6 or len({chunk.chunk_id for chunk in chunks}) != 6:
        raise ValueError("Expected exactly six chunks with distinct IDs.")
    return chunks


def score_chunks(question: str, chunks: list[Chunk]):
    query_words = words(question)
    scored = []
    for chunk in chunks:
        matched = query_words & words(chunk.text)
        scored.append((chunk, len(matched), sorted(matched)))
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to search for (enclose in quotes).")
    parser.add_argument("--k", type=int, default=2, help="Maximum passages to select (default: 2).")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be a positive integer")

    chunks = load_chunks(Path(__file__).resolve().parent / "docs")
    scored = score_chunks(args.question, chunks)
    ranked = sorted(scored, key=lambda item: (-item[1], item[0].chunk_id))

    print(f"Question: {args.question}")
    print(f"k: {args.k}")
    print("\nAll chunk scores (ranked):")
    for chunk, score, matched in ranked:
        print(f"{chunk.chunk_id} | {chunk.filename} | score={score} | matched words: {', '.join(matched) or '(none)'}")

    selected = [item for item in ranked if item[1] > 0][:args.k]
    if not selected:
        print("\nNo matching passages.")
        return
    print("\nSelected passages:")
    for chunk, score, matched in selected:
        print(f"\n{chunk.chunk_id} | {chunk.filename}")
        print(chunk.text)


if __name__ == "__main__":
    main()
