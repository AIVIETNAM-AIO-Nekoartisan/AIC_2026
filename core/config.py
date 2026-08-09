import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "multimodal_products")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "openai/clip-vit-base-patch32")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
VECTOR_DIM = 512  # Kích thước vector của mô hình CLIP
