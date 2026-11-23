"""Default model IDs for retrieval, reranking, and query analysis."""

# Retrieval: strong dense retriever; fits T4 index build (~5–10 min on ~800 docs).
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# T4 Colab: DistilBERT is stable (finite loss); DeBERTa often NaNs on T4 + PyTorch 3.13.
T4_ANALYZER_MODEL = "distilbert-base-uncased"
DEFAULT_ANALYZER_MODEL = T4_ANALYZER_MODEL

# Optional stronger encoder (use on A100 with attn_implementation=eager in code).
ANALYZER_MODEL_DEBERTA = "microsoft/deberta-v3-small"
ANALYZER_MODEL_LARGE = "microsoft/deberta-v3-base"

# Rerank FAISS top-k before temporal scoring (big Contains boost on this corpus).
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Legacy (old artifacts)
LEGACY_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LEGACY_ANALYZER_MODEL = "distilbert-base-uncased"
