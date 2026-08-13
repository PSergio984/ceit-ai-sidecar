"""Shared fixtures: deterministic embedder + temp corpus for fast tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# App modules import Settings() at import time — provide env before imports.
os.environ.setdefault("SIDECAR_TOKEN", "test-token")
os.environ.setdefault("CORPUS_PATH", str(Path("cache").resolve()))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings


class DeterministicEmbedder:
    """Hash-based embedder: same text -> same vector; different texts differ.

    Produces 8-dim normalized vectors — enough to exercise cosine math
    without the ~470 MB SentenceTransformer model.
    """

    def encode(self, texts, normalize_embeddings=True):
        vectors = []
        for text in texts:
            h = 0
            for ch in text:
                h = (h * 31 + ord(ch)) % (2**32 - 1)
            rng = np.random.RandomState(h)
            v = rng.rand(8).astype(np.float32)
            if normalize_embeddings:
                v = v / np.linalg.norm(v)
            vectors.append(v)
        return np.asarray(vectors, dtype=np.float32)


def make_corpus(tmp_path: Path) -> Path:
    """Write a small catalog + policies corpus with realistic doc shapes."""
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True)

    catalog = {
        "source": "academic_papers",
        "schema_version": 1,
        "generated_at": "2026-08-13T02:00:00+00:00",
        "count": 4,
        "documents": [
            {
                "id": "paper-1",
                "corpus": "catalog",
                "title": "Analysis of Groundwater Depletion Caused By Excessive Use of Water Pumps",
                "text": "Analysis of Groundwater Depletion Caused By Excessive Use of Water Pumps. Analysis of Groundwater Depletion Caused By Excessive Use of Water Pumps. authors: Juan Dela Cruz. research_adviser: . technical_adviser: . dean: . department: Civil Engineering. publication_year: 2015. paper_type: Thesis. catalog_code: CEIT-CE-15-014",
                "metadata": {
                    "catalog_code": "CEIT-CE-15-014",
                    "department": "Civil Engineering",
                    "publication_year": 2015,
                    "paper_type": "Thesis",
                    "authors": ["Juan Dela Cruz"],
                },
            },
            {
                "id": "paper-2",
                "corpus": "catalog",
                "title": "Design of a Smart Flood Monitoring System",
                "text": "Design of a Smart Flood Monitoring System. Design of a Smart Flood Monitoring System. authors: Maria Santos. research_adviser: . technical_adviser: . dean: . department: Electrical Engineering. publication_year: 2025. paper_type: Thesis. catalog_code: CEIT-EE-25-01",
                "metadata": {
                    "catalog_code": "CEIT-EE-25-01",
                    "department": "Electrical Engineering",
                    "publication_year": 2025,
                    "paper_type": "Thesis",
                    "authors": ["Maria Santos"],
                },
            },
            {
                "id": "paper-3",
                "corpus": "catalog",
                "title": "Development of a Library Management Mobile Application",
                "text": "Development of a Library Management Mobile Application. Development of a Library Management Mobile Application. authors: Pedro Reyes. research_adviser: . technical_adviser: . dean: . department: Information Technology. publication_year: 2023. paper_type: Capstone. catalog_code: CEIT-IT-23-07",
                "metadata": {
                    "catalog_code": "CEIT-IT-23-07",
                    "department": "Information Technology",
                    "publication_year": 2023,
                    "paper_type": "Capstone",
                    "authors": ["Pedro Reyes"],
                },
            },
            {
                "id": "paper-4",
                "corpus": "catalog",
                "title": "Design of a Solar Powered Air Conditioning System",
                "text": "Design of a Solar Powered Air Conditioning System. Design of a Solar Powered Air Conditioning System. authors: Ana Cruz. research_adviser: . technical_adviser: . dean: . department: Civil Engineering. publication_year: 2022. paper_type: Research. catalog_code: CEIT-CE-22-03",
                "metadata": {
                    "catalog_code": "CEIT-CE-22-03",
                    "department": "Civil Engineering",
                    "publication_year": 2022,
                    "paper_type": "Research",
                    "authors": ["Ana Cruz"],
                },
            },
        ],
    }
    policies = {
        "source": "rulebook",
        "schema_version": 1,
        "generated_at": "2026-08-13T02:00:00+00:00",
        "count": 2,
        "documents": [
            {
                "id": "policy-h1",
                "corpus": "policy",
                "title": "General Information",
                "text": "Section: General Information",
                "metadata": {"policy_type": "header", "header_id": 1, "header_title": "General Information"},
            },
            {
                "id": "policy-h1-r1",
                "corpus": "policy",
                "title": "General Information",
                "text": "Section: General Information\nStudents must present their school ID.",
                "metadata": {"policy_type": "regulation", "header_id": 1, "regulation_id": 1},
            },
        ],
    }
    (corpus / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    (corpus / "policies.json").write_text(json.dumps(policies), encoding="utf-8")
    return corpus


@pytest.fixture
def corpus_path(tmp_path) -> Path:
    return make_corpus(tmp_path)


@pytest.fixture
def deterministic_embedder() -> DeterministicEmbedder:
    return DeterministicEmbedder()


@pytest.fixture
def settings_for(tmp_path, corpus_path) -> Settings:
    return Settings(sidecar_token="test-token", corpus_path=corpus_path, host="127.0.0.1", port=8310)


def embed_from(docs: list[dict]):
    """Deterministic per-document embedder for a known docs list."""

    def embed(docs_):
        vectors = []
        for d in docs_:
            rng = np.random.RandomState(sum(ord(c) for c in d["text"]))
            v = rng.rand(16).astype(np.float32)
            v = v / np.linalg.norm(v)
            vectors.append(v)
        return np.asarray(vectors, dtype=np.float32)

    return embed


def build_test_index(tmp_path: Path, corpus_path: Path, cache_name: str = "cache") -> tuple[Path, list[dict]]:
    """Build a real versioned index with a deterministic embedder; returns (cache, docs)."""
    from app.ingest import load_documents
    from app.rebuild import build_index

    cache = tmp_path / cache_name
    docs = load_documents(corpus_path)
    build_index(corpus_path, "test-model", cache, embed_fn=embed_from(docs))
    return cache, docs
