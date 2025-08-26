# Development Plan for ML Challenge – Text–Image Similarity (Simplified)

## Project Purpose and Goals

From CHALLENGE_BRIEF.md context we need to:
- Answer **only** the three required questions:
  1) Compute a similarity score for each image–text pair and write it to a new CSV column.
  2) Briefly discuss time/memory footprint and optimisation strategies.
  3) Explain packaging/deployment to handle hundreds of millions of daily requests.
- Keep scope minimal and production-sensible: a clean function for scoring, a tiny CLI, and brief docs.

## Context and Background

- Input CSV schema (from the provided file): columns are `url` and `caption` (no `image_name` provided).
- We will implement a small Python module (managed by Poetry) that can be reused inside the existing **text_image_similarity** workspace.
- Defaults will target an open model (e.g., CLIP ViT-B/32) but the plan remains model-agnostic.

## Hard Requirements

- [x] Provide a **single public function** to compute similarity for batches of `(image_url, text)` records.
- [x] Provide a **main script** (in place of a CLI): read `challenge_set.csv`, output `challenge_scored.csv` with a new `similarity` column.
- [x] Robust downloading (timeouts + retries) and graceful error handling (write `error` column when applicable).
- [x] Deterministic defaults (fixed model/version and seeds where applicable).
- [x] Basic logging (counts, durations, failures).
- [x] Unit tests for the public function and CSV pipeline (with tiny offline fixtures).

## Unknowns / Assumptions

- [ ] Model: default to CLIP ViT-B/32 (can be swapped via config).
- [ ] Hardware: CPU-only baseline; GPU optional if available.
- [ ] Image privacy/auth: assume publicly accessible URLs without auth headers.

## Development Phases (Minimal)

### Phase 0 — Context Gathering (no code)
- [x] Read the challenge PDF to confirm exact outputs and any naming constraints.
- [x] Confirm we must preserve input row order and write a `similarity` float column (and optional `error` column).

### Phase 1 — Package Scaffold (Poetry)
- [x] Create subpackage `text_image_similarity/` with `pyproject.toml` (Poetry).
- [x] Add minimal deps: `torch`, `torchvision`, `pillow`, `httpx` (or `requests`), `pandas`, `numpy`, `tqdm`.
- [x] Add dev deps: `pytest`, `pytest-cov`, `ruff`, `black`.
- [x] Configure `__init__.py` and versioning.

### Phase 2 — Core API Design
- [x] Define dataclass `Record(url: str, text: str)`; derive `image_name` internally if needed.
- [x] Define configuration dataclass `Config(batch_size: int, device: str, timeout_s: int, retries: int, precision: str)`.
- [x] Define **public function**:
  - [x] `def score_pairs(records: Iterable[Record], config: Config) -> Iterable[float]` returning scores aligned to input order.
  - [x] Internally: download → preprocess → batch encode (image/text) → cosine similarity.
- [x] Add error policy: on download/processing failure, return `nan` and log reason; caller can persist to `error` column.

### Phase 3 — CSV Pipeline + Main Script
- [x] Implement `score_csv(in_path, out_path, col_url="url", col_text="caption", ...)` streaming in chunks.
- [x] Add main script entry point: `python execute_text_image_similarity_csv.py`.
- [x] Ensure output preserves all original columns plus `similarity` (and `error` when present).

### Phase 4 — Tests (focus on the public function)
- [x] **Unit test (offline)** for `score_pairs` using a stub model to ensure determinism and alignment.
- [x] **Unit test (offline)** for downloader normalization (e.g., rejects huge files, handles bad URLs without hanging).
- [x] **Integration test (offline)** with 2–3 tiny local images and dummy captions to validate end-to-end CSV path.
- [ ] Add coverage target (~80%+) and run in CI locally.

### Phase 5 — Docs (answering the 3 questions)
- [ ] **Question 1** (How): Short README section describing API, CLI, and how the score is computed.
- [ ] **Question 2** (Time/Memory): One concise section listing footprint and concrete optimisations (batching, FP16, caching, streaming, lazy decode, IO concurrency).
- [ ] **Question 3** (Packaging/Deployment at Scale): One concise section with options—container + FastAPI, async inference server, autoscaling on K8s, sharded workers, CDN/cache for images, and observability.

## Refactoring (as needed)
- [ ] If profiling shows hotspots, refactor batching & image preprocessing; consider `torch.compile` (if supported) and memory pinning on GPU.
- [ ] Keep interfaces stable; avoid leaking model-specific types into public API.

## Quality Gates

- [ ] `ruff` and `black` passing; type hints on all public APIs.
- [ ] `pytest -q` passing with adequate coverage.
- [ ] `poetry build` and `poetry check` succeed.
- [ ] CLI smoke test runs and produces `challenge_scored.csv` with 51 rows.

## QA CHECKLIST

- [ ] All three questions addressed explicitly in README.
- [ ] CSV → CSV flow implemented (`similarity` column produced).
- [ ] Error handling validated (bad URLs/timeouts produce `nan` and `error` reason).
- [ ] Deterministic defaults validated.
- [ ] Performance notes documented (time/memory + optimisations).
- [ ] Packaging/deployment notes documented (handles very high QPS via sharding/autoscaling).
- [ ] Code style and tests passing.
