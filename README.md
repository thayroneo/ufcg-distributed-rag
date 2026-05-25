documents.jsonl -> auditoria dos documentos coletados
chunks.jsonl -> resultado das extracoes

- Git LFS - arquivos grandes

esse repositorio usa Git LFS para versionar arquivos grandes, como chunks_all.jsonl e arquivos chunks.jsonl dos nos

depois de clonar o repositorio, rode:

git lfs install
git lfs pull

Sem Git LFS, esses arquivos podem aparecer apenas como ponteiros de texto em vez dos dados completos.

- Embeddings

modelo: intfloat/multilingual-e5-base
dimensao dos vetores: 768
prefixo da consulta: "query: "
normalize_embeddings=True
distancia no Qdrant: cosine

- Como usar os bancos

Opcao 1 - usar o banco pronto

1. baixar e extrair qdrant_storage.zip na raiz do projeto
2. subir os containers: docker compose -f docker-compose.qdrant.yml up -d
3. validar: python scripts/validate_qdrant_dbs.py

Opcao 2 - Recriar o banco a partir dos chunks

1. subir os containers: docker compose -f docker-compose.qdrant.yml up -d
2. popular os bancos: python scripts/build_qdrant_dbs.py
3. validar: python scripts/validate_qdrant_dbs.py

Cada no possui seu proprio banco e uma collection chamada documentos
Portas: eletrica 6333, quimica 6334, computacao 6335, historia 6336, linguagem_ensino 6337