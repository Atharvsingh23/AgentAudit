"""Corpus persistence.

The corpus is generated deterministically, but it also ships as JSONL in
``benchmarks/invoices/``. Both matter for different reasons: the generator
means anyone can scale the corpus up, and the checked-in files mean a
result can be reproduced against exactly the bytes that produced it, even
if the generator changes.

``verify_corpus`` guards that pair — if the generator ever drifts from the
shipped files, CI fails rather than silently invalidating published
numbers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .dataset import build_corpus

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "invoices"
CORPUS_FILE = "corpus.jsonl"
TRUTH_FILE = "ground_truth.jsonl"
MANIFEST_FILE = "manifest.json"


def _digest(records: list[dict[str, Any]]) -> str:
    blob = "\n".join(json.dumps(r, sort_keys=True) for r in records)
    return hashlib.sha256(blob.encode()).hexdigest()


def export_corpus(
    n: int = 100,
    seed: int = 7,
    directory: str | Path = DEFAULT_DIR,
) -> Path:
    """Write the corpus, ground truth and a manifest to disk."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    documents = build_corpus(n=n, seed=seed)

    corpus_records = [
        {
            "document_id": d["document_id"],
            "raw_text": d["raw_text"],
            "fields": d["fields"],
        }
        for d in documents
    ]
    truth_records = [
        {"document_id": d["document_id"], "truth": d["truth"]} for d in documents
    ]

    with (directory / CORPUS_FILE).open("w") as fh:
        for record in corpus_records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    with (directory / TRUTH_FILE).open("w") as fh:
        for record in truth_records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    manifest = {
        "n": n,
        "seed": seed,
        "generator": "agentaudit.dataset.build_corpus",
        "corpus_sha256": _digest(corpus_records),
        "truth_sha256": _digest(truth_records),
        "schema": "invoice",
        "note": (
            "Synthetic. Regenerate with `python -m agentaudit.corpus export`. "
            "Hashes are checked by tests to catch generator drift."
        ),
    }
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    return directory


def load_corpus(directory: str | Path = DEFAULT_DIR) -> list[dict[str, Any]]:
    """Load the shipped corpus from disk, rejoined with ground truth."""
    directory = Path(directory)
    corpus_path = directory / CORPUS_FILE
    truth_path = directory / TRUTH_FILE
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} not found; run `python -m agentaudit.corpus export`"
        )

    truth_by_id = {
        json.loads(line)["document_id"]: json.loads(line)["truth"]
        for line in truth_path.read_text().splitlines() if line.strip()
    }

    documents = []
    for line in corpus_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record["truth"] = truth_by_id[record["document_id"]]
        documents.append(record)
    return documents


def verify_corpus(directory: str | Path = DEFAULT_DIR) -> bool:
    """Check the shipped files still match what the generator produces."""
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST_FILE).read_text())
    documents = build_corpus(n=manifest["n"], seed=manifest["seed"])

    corpus_records = [
        {"document_id": d["document_id"], "raw_text": d["raw_text"],
         "fields": d["fields"]}
        for d in documents
    ]
    truth_records = [
        {"document_id": d["document_id"], "truth": d["truth"]} for d in documents
    ]
    return (
        _digest(corpus_records) == manifest["corpus_sha256"]
        and _digest(truth_records) == manifest["truth_sha256"]
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        ok = verify_corpus()
        print("corpus matches generator" if ok else "CORPUS DRIFT DETECTED")
        sys.exit(0 if ok else 1)

    path = export_corpus()
    print(f"wrote corpus to {path}")
