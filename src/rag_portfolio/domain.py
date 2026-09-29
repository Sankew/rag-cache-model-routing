"""Small immutable contracts shared by every portfolio extension."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant_id: str
    roles: frozenset[str]
    clearance: int = 0


@dataclass(frozen=True)
class ACL:
    tenant_id: str
    allowed_roles: frozenset[str]
    allowed_users: frozenset[str]
    classification: int = 0
    revision: int = 1


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    acl: ACL
    version: int
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class Citation:
    document_id: str
    chunk_id: str
    version: int


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
