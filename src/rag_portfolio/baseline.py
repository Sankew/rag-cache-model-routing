"""Offline ingestion, retrieval, and extractive generation for the demo."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Callable, Iterable, Protocol

from .domain import ACL, Answer, Chunk, Citation, ScoredChunk


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Split on words; size is an approximation of model tokens."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("require size > overlap >= 0")
    words = text.split()
    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


class Embedder(Protocol):
    def embed(self, text: str) -> tuple[float, ...]: ...


class HashingEmbedder:
    """Deterministic lexical embedding for offline tests and local demos."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    def embed(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self.dimensions
        for token in tokens(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            number = int.from_bytes(digest, "big")
            values[number % self.dimensions] += 1.0
        magnitude = math.sqrt(sum(value * value for value in values))
        return tuple(value / magnitude for value in values) if magnitude else tuple(values)


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions differ")
    return sum(a * b for a, b in zip(left, right))


class MemoryIndex:
    """Deterministic vector index. Production adapters can expose same query API."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._document_versions: dict[tuple[str, str], int] = {}
        self.revision = 0

    def upsert(self, chunks: Iterable[Chunk]) -> None:
        batch = list(chunks)
        if not batch:
            return
        by_document: dict[tuple[str, str], list[Chunk]] = {}
        for chunk in batch:
            if not chunk.acl.tenant_id:
                raise ValueError("tenant_id is required on every chunk")
            key = (chunk.acl.tenant_id, chunk.document_id)
            by_document.setdefault(key, []).append(chunk)
            existing = self._chunks.get(chunk.chunk_id)
            if existing is not None and (existing.acl.tenant_id, existing.document_id) != key:
                raise ValueError("chunk ID collision across documents or tenants")
        for key, items in by_document.items():
            versions = {chunk.version for chunk in items}
            if len(versions) != 1:
                raise ValueError("one document batch cannot mix versions")
            version = next(iter(versions))
            if version < self._document_versions.get(key, 0):
                raise ValueError("cannot ingest an older document version")
        for key, items in by_document.items():
            self._chunks = {
                chunk_id: old for chunk_id, old in self._chunks.items()
                if (old.acl.tenant_id, old.document_id) != key
            }
            for chunk in items:
                self._chunks[chunk.chunk_id] = chunk
            self._document_versions[key] = items[0].version
        self.revision += 1

    def delete_document(self, tenant_id: str, document_id: str) -> None:
        key = (tenant_id, document_id)
        before = len(self._chunks)
        self._chunks = {
            chunk_id: chunk for chunk_id, chunk in self._chunks.items()
            if (chunk.acl.tenant_id, chunk.document_id) != key
        }
        self._document_versions.pop(key, None)
        if len(self._chunks) != before:
            self.revision += 1

    def query(
        self,
        vector: tuple[float, ...],
        limit: int = 5,
        predicate: Callable[[Chunk], bool] | None = None,
    ) -> list[ScoredChunk]:
        if limit < 1:
            raise ValueError("limit must be positive")
        candidates = (
            ScoredChunk(chunk, cosine(vector, chunk.embedding))
            for chunk in self._chunks.values()
            if predicate is None or predicate(chunk)
        )
        return sorted(candidates, key=lambda hit: (-hit.score, hit.chunk.chunk_id))[:limit]

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def all_chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks.values())


def ingest_document(
    document_id: str,
    text: str,
    acl: ACL,
    embedder: Embedder,
    *,
    version: int = 1,
    size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    if not document_id or not acl.tenant_id:
        raise ValueError("document_id and tenant_id are required")
    parts = chunk_text(text, size, overlap)
    if not parts:
        raise ValueError("cannot ingest an empty document")
    return [
        Chunk(f"{acl.tenant_id}:{document_id}:v{version}:{i}", f"{acl.tenant_id}:{document_id}", part, acl, version, embedder.embed(part))
        for i, part in enumerate(parts)
    ]


class Generator(Protocol):
    model_id: str

    def generate(self, question: str, evidence: list[ScoredChunk]) -> Answer: ...


class ExtractiveDemoGenerator:
    """Offline smoke-test generator; outputs a source sentence, never a CV metric."""

    model_id = "extractive-demo"

    def generate(self, question: str, evidence: list[ScoredChunk]) -> Answer:
        if not evidence:
            return Answer("I don't know based on the available documents.", (), self.model_id)
        stopwords = {"a", "an", "the", "is", "are", "do", "does", "how", "what", "when", "where", "who", "many", "of", "in", "on", "to", "for"}
        question_terms = set(tokens(question)) - stopwords
        best: tuple[int, float, Chunk, str] | None = None
        for hit in evidence:
            for sentence in re.split(r"(?<=[.!?])\s+", hit.chunk.text):
                overlap = len(question_terms.intersection(tokens(sentence)))
                value = (overlap, hit.score)
                if best is None or value > best[:2]:
                    best = (overlap, hit.score, hit.chunk, sentence)
        assert best is not None
        if best[0] < 2:
            return Answer("I don't know based on the available documents.", (), self.model_id)
        chunk = best[2]
        return Answer(best[3], (Citation(chunk.document_id, chunk.chunk_id, chunk.version),), self.model_id)
