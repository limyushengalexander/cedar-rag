# PS5 Cedar Lab RAG Assignment

Author: Alex Lim.

Completed individually by Alex Lim.

## Environment and setup

Model: `sentence-transformers/all-MiniLM-L6-v2`, run locally on CPU. No external
model API, API key, vector database, or web interface is used.

Recorded versions, also verified against the existing `.venv`:

| Component | Version |
| --- | --- |
| Python | 3.11.9 |
| sentence-transformers | 6.1.0 |
| numpy | 2.4.6 |
| torch | 2.14.1 |

The existing environment is ready. For a fresh setup on this Windows computer,
use the installed Python 3.11 interpreter below to create `.venv`. The base
interpreter is needed only to create the environment; subsequent commands use
the virtual environment directly. Do not recreate the current environment for
submission checking.

```powershell
Set-Location -LiteralPath 'C:\Users\limyu\OneDrive\Desktop\MIT SDM and CEE =)\Info\Course Selection\Fall Term 26_27\1.125\Problem Set 5\cedar-rag'
& 'C:\Users\limyu\AppData\Local\Python\pythoncore-3.11-64\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install "sentence-transformers==6.1.0" "numpy==2.4.6" "torch==2.14.1"
.\.venv\Scripts\python.exe -m pip check
```

This pins the three recorded packages; their transitive dependencies are
resolved by pip rather than locked individually. On another Windows computer,
replace the base interpreter path with that computer's Python 3.11.9 path.

The embedding scripts load only from the local cache. After a fresh install,
download the public model once with the following setup command (requires
internet access; no API key). It creates the cache used by the offline scripts.

```powershell
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu', cache_folder='.venv/model-cache'); print('Model loaded on', model.device)"
```

The installed environment and downloaded cache are excluded from Git. Source
documents, code, recorded versions, results, evidence, and the reflection are
included in the submission.

## Task 1: keyword retrieval

Run these commands in Windows PowerShell. The existing virtual environment is
used directly; activation and additional package installation are unnecessary.

```powershell
Set-Location -LiteralPath 'C:\Users\limyu\OneDrive\Desktop\MIT SDM and CEE =)\Info\Course Selection\Fall Term 26_27\1.125\Problem Set 5\cedar-rag'
.\.venv\Scripts\python.exe .\keyword_retriever.py "May a trained graduate student operate the laser alone at 19:00?" --k 2
```

Omit `--k 2` to use the default of two passages. Change the quoted question and
`--k` value for other searches; `k` must be a positive integer.

To save Q1's output again:

```powershell
New-Item -ItemType Directory -Force -Path .\evidence | Out-Null
.\.venv\Scripts\python.exe .\keyword_retriever.py "May a trained graduate student operate the laser alone at 19:00?" --k 2 | Tee-Object -FilePath .\evidence\task1-keyword-q1.txt
```

`load_chunks` reads the three source files and separates each label from its
passage, retaining the ID, filename, and complete text. There are six chunks.
`words` lowercases text, extracts alphanumeric words, and removes exactly the
specified 17 stopwords. Sets discard repeated words. `score_chunks` intersects
the question's word set with each passage's word set; the intersection size is
the score. Labels and filenames never enter the search.

`main` sorts by descending score, then ascending ID, prints all six scores and
their matched words, and displays up to `k` passages with positive scores. If
all scores are zero, it prints `No matching passages.` This is literal keyword
retrieval with no stemming, synonyms, embeddings, or answer generation.

## Task 2: CPU embedding retrieval

Run Q1 and save the actual inspection output in Windows PowerShell:

The `Tee-Object -FilePath` commands shown for Tasks 1 and 2 replace the named
evidence file. To inspect a question without replacing evidence, run the same
Python command without its `| Tee-Object ...` suffix.

```powershell
Set-Location -LiteralPath 'C:\Users\limyu\OneDrive\Desktop\MIT SDM and CEE =)\Info\Course Selection\Fall Term 26_27\1.125\Problem Set 5\cedar-rag'
New-Item -ItemType Directory -Force -Path .\evidence | Out-Null
.\.venv\Scripts\python.exe .\embedding_retriever.py "May a trained graduate student operate the laser alone at 19:00?" --k 2 | Tee-Object -FilePath .\evidence\embedding-inspection.txt
```

Omit `--k 2` to use the default. Positive values larger than six return all six
available passages. Additional individually quoted questions can follow the
first question in the same command; the document matrix is computed once per
run and reused for all questions.

The script reuses Task 1's paragraph loader without modifying it. The existing
local model cache under `.venv/model-cache` is required; `local_files_only=True`
keeps model loading offline. Both encoding methods use the same
`sentence-transformers/all-MiniLM-L6-v2` model on CPU. `encode_document` receives
only passage text, while `encode_query` receives each complete question. Both
request normalized NumPy embeddings. IDs and filenames remain separate metadata.

The resulting matrix has six rows of 384 coordinates. The script preserves all
coordinates and converts to float64 for subsequent arithmetic. It checks that
each vector's Euclidean length is within 0.000001 of one. Matrix multiplication
(`documents @ query`) computes one dot product per passage using all 384
coordinates. For unit vectors this agrees with cosine similarity. Sorting uses
the unrounded score descending, then the chunk ID ascending. Selection always
returns the first `k` available passages without a score threshold.

The inspection prints the first 12 coordinates, all lengths, all six scores,
and complete selected passages. For the highest-ranked passage it multiplies
matching coordinates, sums the first 12 and remaining 372 products separately,
and checks the full score against cosine computed by dividing the dot product
by both vector lengths. Only displayed numbers are rounded. Coordinates are
not assigned semantic labels. This task retrieves passages; it generates no answer.

## Task 3: recorded runs and completed judgments

**Rerun warning:** `run_task3.py` opens `results.csv`,
`evidence/retrieval-runs.txt`, and `evidence/k-experiment.txt` for writing.
Rerunning it overwrites all three files, resets both evidence-complete columns
and the observation column to blank, and removes the appended k-experiment
interpretation. The submitted CSV already contains completed judgments, and
the k-experiment file includes the supplied interpretation. Preserve these
files before any future rerun, or run a copy of the project. Do not run the
following command as a submission check.

Run the four specified questions with both unchanged retrievers at `k=2`, then
the Q1 embedding experiment at `k=1`, `k=2`, and `k=3`:

```powershell
Set-Location -LiteralPath 'C:\Users\limyu\OneDrive\Desktop\MIT SDM and CEE =)\Info\Course Selection\Fall Term 26_27\1.125\Problem Set 5\cedar-rag'
.\.venv\Scripts\python.exe .\run_task3.py
```

This writes `evidence/retrieval-runs.txt`, `results.csv`, and
`evidence/k-experiment.txt` and prints compact comparison tables. Document
embeddings are computed once and reused throughout the run. The runner calls
the existing keyword CLI and embedding inspection function; CSV selections
and scores are extracted from their actual printed output in rank order.
Semicolons separate corresponding selected IDs and scores in CSV cells.
The runner initially leaves both evidence-complete columns and the observation
column blank; the submitted file has since been completed with the supplied
judgments. Q4's required evidence is recorded as `absent`; its nearest
passages are not marked as successful answer retrieval. No answers or Task 4
work are generated. Rerunning replaces these three Task 3 output files.

## Task 4 evidence and individual reflection

`evidence/answer-audit-q1.md` preserves the original prompt, original answer,
audit, single added revision instruction, revised answer, and final audit.
`evidence/answer-audit-q4.md` preserves its prompt, answer, and passing audit.
These supplied answers were generated in separate fresh ChatGPT conversations
using only the selected passages, as reported by the user. They were recorded
verbatim rather than regenerated by this project. `reflection.md` contains
Alex Lim's supplied individual reflection verbatim.

## Submission files

- `docs/access.txt`, `docs/training.txt`, `docs/booking.txt`
- `keyword_retriever.py`, `embedding_retriever.py`, `run_task3.py`
- `setup-versions.txt`, `README.md`, `.gitignore`, `results.csv`
- `evidence/embedding-inspection.txt`, `evidence/retrieval-runs.txt`
- `evidence/k-experiment.txt`, `evidence/answer-audit-q1.md`
- `evidence/answer-audit-q4.md`, `evidence/task1-keyword-q1.txt`
- `reflection.md`
