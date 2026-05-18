
documents.jsonl -> auditoria dos documentos coletados
chunks.jsonl -> resultado das extrações

- Embeddings

modelo: intfloat/multilingual-e5-base
dimensão dos vetores: 768
prefixo da consulta: "query: "
normalize_embeddings=True
distância no Qdrant: cosine

- Como usar os bancos

Opção 1 - usar o banco pronto

1. baixar e extrair qdrant_storage.zip na raiz do projeto
2. subir os containers: docker compose -f docker-compose.qdrant.yml up -d
3. validar: python scripts/validate_qdrant_databases.py

Opção 2 — Recriar o banco a partir dos chunks

1. subir os containers: docker compose -f docker-compose.qdrant.yml up -d
2. popular os bancos: python scripts/build_qdrant_databases.py
3. validar: python scripts/validate_qdrant_databases.py

Cada nó possui seu próprio banco e uma collection chamada documentos
Portas: eletrica 6333, quimica 6334, computacao 6335