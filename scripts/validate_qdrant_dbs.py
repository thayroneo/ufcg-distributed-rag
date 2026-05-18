import sys
from typing import Any, Dict, List
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch

COLLECTION_NAME = "documentos"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
QUERY_PREFIX = "query: "
TOP_K = 5

NODES: Dict[str, Dict[str, str]] = {
    "eletrica": {"label": "Engenharia Elétrica", "url": "http://localhost:6333"},
    "quimica": {"label": "Engenharia Química", "url": "http://localhost:6334"},
    "computacao": {"label": "Computação", "url": "http://localhost:6335"},
}

DEFAULT_QUERY = "redes neurais"

def detect_device() -> str:
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

def query_qdrant(client: QdrantClient, query_vector: List[float], top_k: int):
    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return response.points
    except AttributeError:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

def clip(text: Any, n: int = 280) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."

def main() -> None:
    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY

    print("Consulta:", query)
    print("Modelo:", EMBEDDING_MODEL_NAME)

    device = detect_device()
    print("Device:", device)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    query_vector = model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    for node_name, cfg in NODES.items():
        print("\n" + "-" * 80)
        print(f"NÓ: {node_name} — {cfg['label']} | {cfg['url']}")
        print("-" * 80)

        client = QdrantClient(url=cfg["url"])
        info = client.get_collection(COLLECTION_NAME)
        print("Status:", getattr(info, "status", None))
        print("Points:", getattr(info, "points_count", None))

        results = query_qdrant(client, query_vector, TOP_K)
        if not results:
            print("Nenhum resultado encontrado.")
            continue

        for i, hit in enumerate(results, start=1):
            payload = hit.payload or {}
            score = getattr(hit, "score", None)
            print(f"\n[{i}] score={score}")
            print("title:", payload.get("title"))
            print("year:", payload.get("year"), "| pages:", payload.get("page_start"), "-", payload.get("page_end"))
            print("chunk_id:", payload.get("chunk_id"))
            print("text:", clip(payload.get("text")))

if __name__ == "__main__":
    main()
