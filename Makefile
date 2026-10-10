.PHONY: demo eval test semantic

demo:
	uv run rag demo

eval:
	uv run rag eval --sweep

semantic:
	uv sync --extra embeddings
	RAG_EMBEDDER=bge-small RAG_RERANKER=cross-encoder uv run rag eval --rerank --sweep

test:
	uv run pytest
