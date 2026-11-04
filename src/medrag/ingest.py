"""Ingestion: download → hash check → extract → chunk → embed → Qdrant.

Run with `python -m medrag.ingest`. Safe to re-run: point IDs are deterministic and
each document's old points are replaced, so no duplicates are created.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pymupdf
import yaml
from qdrant_client import QdrantClient, models

from medrag.config import Settings, get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

EMBEDDING_DIM = 1024  # bge-m3 dense vector size
UPSERT_BATCH = 128
_ID_NAMESPACE = uuid.UUID("3f6c2a1e-8d4b-4c1a-9a57-6d2f0b7e5c11")


@dataclass(frozen=True)
class Source:
    doc_id: str
    title: str
    url: str
    language: str
    licence: str
    sha256: str


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    language: str
    page: int
    chunk_id: str
    text: str

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(_ID_NAMESPACE, self.chunk_id))


def load_manifest(path: Path) -> list[Source]:
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))["documents"]
    names = [f.name for f in fields(Source)]
    return [Source(**{name: entry[name] for name in names}) for entry in entries]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(source: Source, raw_dir: Path, timeout: float) -> Path:
    """Download a PDF once and verify its hash. Raises on a hash mismatch."""
    target = raw_dir / f"{source.doc_id}.pdf"
    if target.exists() and sha256_file(target) == source.sha256:
        return target
    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    with httpx.stream("GET", source.url, follow_redirects=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for block in response.iter_bytes():
                f.write(block)
    digest = sha256_file(tmp)
    if digest != source.sha256:
        tmp.unlink()
        raise ValueError(f"{source.doc_id}: sha256 mismatch (got {digest})")
    tmp.replace(target)
    return target


def clean_text(text: str) -> str:
    text = re.sub(r"­\s*", "", text)  # soft hyphens from line-break hyphenation
    return re.sub(r"\s+", " ", text).strip()


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return (1-based page number, cleaned text) for every non-empty page."""
    with pymupdf.open(path) as doc:
        pages = [(number, clean_text(page.get_text())) for number, page in enumerate(doc, 1)]
    return [(number, text) for number, text in pages if text]


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into ~`size`-character windows with `overlap`, cutting at spaces."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            end = space if space != -1 else end
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
        space = text.find(" ", start, end)
        if text[start - 1] != " " and space != -1:
            start = space + 1
    return [chunk for chunk in chunks if chunk]


def build_chunks(source: Source, pages: list[tuple[int, str]], settings: Settings) -> list[Chunk]:
    """Chunk page by page, so every chunk maps to exactly one citable page."""
    return [
        Chunk(
            doc_id=source.doc_id,
            title=source.title,
            language=source.language,
            page=number,
            chunk_id=f"{source.doc_id}:p{number}:{index}",
            text=piece,
        )
        for number, text in pages
        for index, piece in enumerate(chunk_text(text, settings.chunk_size, settings.chunk_overlap))
    ]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_embedder(settings: Settings) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    device = resolve_device(settings.device)
    model = SentenceTransformer(settings.embedding_model, device=device)
    if device != "cpu":
        model.half()  # fp16 halves memory, important on 8 GB machines
    model.max_seq_length = 512  # chunks are ~250 tokens; keeps memory bounded
    return model


def embed(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    vectors = model.encode(texts, batch_size=16, normalize_embeddings=True)
    return vectors.astype("float32").tolist()


def ensure_collection(client: QdrantClient, name: str) -> None:
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
        )


def replace_document(
    client: QdrantClient, collection: str, chunks: list[Chunk], vectors: list[list[float]]
) -> None:
    """Delete a document's old points, then upsert the new ones."""
    doc_filter = models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=chunks[0].doc_id))]
    )
    client.delete(collection, points_selector=models.FilterSelector(filter=doc_filter), wait=True)
    points = [
        models.PointStruct(id=chunk.point_id, vector=vector, payload=asdict(chunk))
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    for i in range(0, len(points), UPSERT_BATCH):
        client.upsert(collection, points=points[i : i + UPSERT_BATCH], wait=True)


def run(settings: Settings) -> None:
    sources = load_manifest(settings.manifest_path)
    client = QdrantClient(url=settings.qdrant_url, timeout=int(settings.request_timeout_s))
    ensure_collection(client, settings.qdrant_collection)
    model = load_embedder(settings)
    for source in sources:
        path = download(source, settings.raw_dir, settings.request_timeout_s)
        chunks = build_chunks(source, extract_pages(path), settings)
        vectors = embed(model, [chunk.text for chunk in chunks])
        replace_document(client, settings.qdrant_collection, chunks, vectors)
        log.info("%-36s %4d chunks", source.doc_id, len(chunks))
    total = client.count(settings.qdrant_collection).count
    log.info("Collection '%s' now holds %d points.", settings.qdrant_collection, total)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run(get_settings())


if __name__ == "__main__":
    main()
