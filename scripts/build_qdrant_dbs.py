import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
import torch

# CONFIGS

PROJECT_ROOT = Path(".")
COLLECTION_NAME = "documentos"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_VECTOR_SIZE = 768
EMBEDDING_DISTANCE = Distance.COSINE
PASSAGE_PREFIX = "passage: "
BATCH_SIZE = 32
NORMALIZE_EMBEDDINGS = True
RECREATE_COLLECTIONS = True
CREATE_PAYLOAD_INDEXES = True

NODES: Dict[str, Dict[str, Any]] = {
    "eletrica": {
        "label": "Engenharia Elétrica",
        "url": "http://localhost:6333",
        "chunks_path": PROJECT_ROOT / "nodes" / "eletrica" / "chunks.jsonl",
        "storage_path": PROJECT_ROOT / "qdrant_storage" / "eletrica",
    },
    "quimica": {
        "label": "Engenharia Química",
        "url": "http://localhost:6334",
        "chunks_path": PROJECT_ROOT / "nodes" / "quimica" / "chunks.jsonl",
        "storage_path": PROJECT_ROOT / "qdrant_storage" / "quimica",
    },
    "computacao": {
        "label": "Computação",
        "url": "http://localhost:6335",
        "chunks_path": PROJECT_ROOT / "nodes" / "computacao" / "chunks.jsonl",
        "storage_path": PROJECT_ROOT / "qdrant_storage" / "computacao",
    },
    "historia": {
        "label": "Historia",
        "url": "http://localhost:6336",
        "chunks_path": PROJECT_ROOT / "nodes" / "historia" / "chunks.jsonl",
        "storage_path": PROJECT_ROOT / "qdrant_storage" / "historia",
    },
    "linguagem_ensino": {
        "label": "Linguagem e Ensino",
        "url": "http://localhost:6337",
        "chunks_path": PROJECT_ROOT / "nodes" / "linguagem_ensino" / "chunks.jsonl",
        "storage_path": PROJECT_ROOT / "qdrant_storage" / "linguagem_ensino",
    },
}

PAYLOAD_INDEXES = {
    "node": PayloadSchemaType.KEYWORD,
    "doc_id": PayloadSchemaType.KEYWORD,
    "chunk_id": PayloadSchemaType.KEYWORD,
    "department": PayloadSchemaType.KEYWORD,
    "program": PayloadSchemaType.KEYWORD,
    "document_type": PayloadSchemaType.KEYWORD,
    "collection_set_spec": PayloadSchemaType.KEYWORD,
    "collection_name": PayloadSchemaType.KEYWORD,
    "year": PayloadSchemaType.INTEGER,
}

# Funções de apoio

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido em {path}, linha {line_no}: {exc}") from exc

def stable_qdrant_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

def chunk_to_payload(chunk: Dict[str, Any]) -> Dict[str, Any]:
    keep_fields = [
        "chunk_id",
        "doc_id",
        "node",
        "text",
        "chunk_index",
        "chunk_chars",
        "chunk_sha256",
        "page_start",
        "page_end",
        "title",
        "authors",
        "contributors",
        "abstract",
        "keywords",
        "year",
        "dc_date",
        "department",
        "program",
        "document_type",
        "handle",
        "item_url",
        "pdf_url",
        "source",
        "oai_identifier",
        "record_set_specs",
        "collection_set_spec",
        "collection_name",
        "pdf_sha256",
        "text_sha256",
        "extraction_backend",
    ]

    payload = {key: chunk.get(key) for key in keep_fields if key in chunk}

    payload["embedding_model"] = EMBEDDING_MODEL_NAME
    payload["embedding_prefix"] = PASSAGE_PREFIX.strip()
    payload["embedding_vector_size"] = EMBEDDING_VECTOR_SIZE
    payload["embedding_distance"] = "cosine"
    payload["embedding_normalized"] = NORMALIZE_EMBEDDINGS

    return payload

def detect_device() -> str:
    if torch is not None and torch.cuda.is_available():
        try:
            print(f"CUDA disponível: {torch.cuda.get_device_name(0)}")
        except Exception:
            print("CUDA disponível")
        return "cuda"
    print("CUDA não disponível, usando CPU")
    return "cpu"

def batched(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]

def load_chunks(path: Path) -> List[Dict[str, Any]]:
    chunks = list(read_jsonl(path))
    if not chunks:
        raise ValueError(f"Nenhum chunk encontrado em: {path}")
    missing_text = [c.get("chunk_id", "sem_id") for c in chunks if not c.get("text")]
    if missing_text:
        raise ValueError(f"Há chunks sem campo text em {path}. Exemplos: {missing_text[:5]}")
    return chunks

# QDRANT

def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    try:
        return client.collection_exists(collection_name)
    except AttributeError:
        names = [c.name for c in client.get_collections().collections]
        return collection_name in names

def create_or_reset_collection(client: QdrantClient) -> None:
    if collection_exists(client, COLLECTION_NAME):
        if RECREATE_COLLECTIONS:
            print(f"  - removendo collection existente: {COLLECTION_NAME}")
            client.delete_collection(COLLECTION_NAME)
        else:
            print(f"  - collection já existe, mantendo: {COLLECTION_NAME}")
            return

    print(f"  - criando collection: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_VECTOR_SIZE,
            distance=EMBEDDING_DISTANCE,
        ),
    )

def create_payload_indexes(client: QdrantClient) -> None:
    if not CREATE_PAYLOAD_INDEXES:
        return

    print("  - criando índices de payload para filtros comuns")
    for field_name, schema in PAYLOAD_INDEXES.items():
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema,
            )
        except (UnexpectedResponse, Exception) as exc:
            print(f"    Não foi possível criar índice '{field_name}': {type(exc).__name__}")

def upsert_chunks(
    client: QdrantClient,
    model: SentenceTransformer,
    chunks: List[Dict[str, Any]],
    node_name: str,
) -> int:
    total = 0

    for batch in batched(chunks, BATCH_SIZE):
        texts = [PASSAGE_PREFIX + str(chunk["text"]) for chunk in batch]
        vectors = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            show_progress_bar=False,
        )

        points = []
        for chunk, vector in zip(batch, vectors):
            chunk_id = str(chunk["chunk_id"])
            points.append(
                PointStruct(
                    id=stable_qdrant_id(chunk_id),
                    vector=vector.tolist(),
                    payload=chunk_to_payload(chunk),
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=False,
        )

        total += len(points)
        print(f"    {node_name}: {total}/{len(chunks)} chunks inseridos")

    return total

def populate_node(
    node_name: str,
    cfg: Dict[str, Any],
    model: SentenceTransformer,
) -> Dict[str, Any]:
    print(f"\nPOPULANDO NÓ: {node_name} — {cfg['label']}")
    print("URL:", cfg["url"])
    print("Chunks:", cfg["chunks_path"])

    chunks = load_chunks(cfg["chunks_path"])
    client = QdrantClient(url=cfg["url"], timeout=120, check_compatibility=False,)
    create_or_reset_collection(client)
    create_payload_indexes(client)
    inserted = upsert_chunks(client, model, chunks, node_name)
    info = client.get_collection(COLLECTION_NAME)
    vectors_count = getattr(info, "vectors_count", None)
    points_count = getattr(info, "points_count", None)

    summary = {
        "node": node_name,
        "label": cfg["label"],
        "url": cfg["url"],
        "collection_name": COLLECTION_NAME,
        "chunks_path": str(cfg["chunks_path"]),
        "storage_path": str(cfg["storage_path"]),
        "chunks_read": len(chunks),
        "points_inserted": inserted,
        "qdrant_vectors_count": vectors_count,
        "qdrant_points_count": points_count,
    }
    print(f"Resumo {node_name}: {summary}")
    return summary

def write_manifest(summaries: List[Dict[str, Any]], device: str) -> None:
    manifest = {
        "created_at": now(),
        "scope": "vector_databases_only",
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_model_note": "Use prefixo 'query: ' para consultas e 'passage: ' para documentos/chunks",
        "embedding_vector_size": EMBEDDING_VECTOR_SIZE,
        "embedding_distance": "cosine",
        "embedding_normalized": NORMALIZE_EMBEDDINGS,
        "device_used": device,
        "batch_size": BATCH_SIZE,
        "nodes": summaries,
        "payload_index_fields": list(PAYLOAD_INDEXES.keys()) if CREATE_PAYLOAD_INDEXES else [],
    }
    path = PROJECT_ROOT / "manifest_qdrant.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nManifest:", path)

def main() -> None:
    print("\nColeção:", COLLECTION_NAME)
    print("Embedding model:", EMBEDDING_MODEL_NAME)
    print("Vector size:", EMBEDDING_VECTOR_SIZE)
    print("Distância:", "cosine")

    device = detect_device()
    print("\nCarregando modelo de embedding")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    summaries = []
    for node_name, cfg in NODES.items():
        summaries.append(populate_node(node_name, cfg, model))

    write_manifest(summaries, device)

    print("Eng. Elétrica: http://localhost:6333/dashboard")
    print("Eng. Química: http://localhost:6334/dashboard")
    print("Computação: http://localhost:6335/dashboard")
    print("Historia: http://localhost:6336/dashboard")
    print("Linguagem e Ensino: http://localhost:6337/dashboard")

if __name__ == "__main__":
    main()
