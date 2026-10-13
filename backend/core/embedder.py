import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from backend.config import AVAILABLE_MODELS, DEFAULT_MODEL_KEY, CACHE_DIR, get_logger

logger = get_logger("embedder")

class EmbedderManager:
    """
    Manages SentenceTransformer embedding models with lazy loading and disk caching.
    """
    def __init__(self):
        self._models: Dict[str, SentenceTransformer] = {}
        self.cache_dir = CACHE_DIR

    def get_model(self, model_key: str = DEFAULT_MODEL_KEY) -> SentenceTransformer:
        if model_key not in AVAILABLE_MODELS:
            model_key = DEFAULT_MODEL_KEY

        if model_key not in self._models:
            model_meta = AVAILABLE_MODELS[model_key]
            model_name = model_meta["name"]
            logger.info(f"Loading embedding model '{model_name}' on CPU...")
            
            # Configure torch CPU threads
            torch.set_num_threads(4)
            model = SentenceTransformer(model_name, device="cpu")
            self._models[model_key] = model
            logger.info(f"Model '{model_key}' successfully loaded.")

        return self._models[model_key]

    def _compute_cache_key(self, texts: List[str], model_key: str) -> str:
        content_repr = f"{model_key}::" + "||".join(texts)
        return hashlib.sha256(content_repr.encode("utf-8")).hexdigest()

    def embed_chunks(
        self,
        chunks: List[Dict],
        model_key: str = DEFAULT_MODEL_KEY,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Embeds a list of chunks, loading from disk cache if already computed.
        Returns a normalized 2D numpy array of shape (N, dim).
        """
        texts = [c.get("semantic_metadata_text", c["text"]) for c in chunks]
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        cache_key = self._compute_cache_key(texts, model_key)
        cache_path = self.cache_dir / f"{cache_key}.npz"

        if use_cache and cache_path.is_file():
            try:
                data = np.load(cache_path)
                logger.info(f"Loaded {len(texts)} chunk embeddings from disk cache ({cache_key[:8]}).")
                return data["embeddings"]
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # Compute embeddings on CPU
        model = self.get_model(model_key)
        embeddings = model.encode(
            texts,
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # Save to cache
        try:
            np.savez_compressed(cache_path, embeddings=embeddings)
            logger.info(f"Saved embeddings to cache ({cache_key[:8]}).")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

        return embeddings

    def embed_query(self, query: str, model_key: str = DEFAULT_MODEL_KEY) -> np.ndarray:
        """
        Embeds a single query string into a normalized 1D vector (dim,).
        """
        model = self.get_model(model_key)
        vector = model.encode(
            query.strip(),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return vector

embedder_manager = EmbedderManager()
