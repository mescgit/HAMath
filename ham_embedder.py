"""
HAMesh Embedder — LLM-free embedding via sentence-transformers

Replaces the Bonsai/llama.cpp embedding dependency for HAMesh 2.0.
All embeddings are produced locally by a frozen sentence-transformer model.
No server, no API calls, no generation, no hallucination.

The model is loaded once and reused. Default: all-mpnet-base-v2 (768-dim).
This is a drop-in replacement for the embed() call in ham_brain.py.

Usage:
    from ham_embedder import Embedder
    emb = Embedder()          # loads model on first call
    vec = emb.embed("Fermat's Last Theorem")  # returns torch.Tensor (768,)
    vecs = emb.embed_batch(["foo", "bar"])    # returns torch.Tensor (2, 768)
"""

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


# Default model — 768-dim, strong semantic quality, runs on CPU or GPU
DEFAULT_MODEL = "all-mpnet-base-v2"


class Embedder:
    """
    Wraps a sentence-transformer model with the same interface as
    ham_brain.embed() so the rest of the codebase needs no changes.

    The model is frozen — no training, no fine-tuning.
    Embeddings are deterministic: same text always produces same vector.
    """

    def __init__(self, model_name=DEFAULT_MODEL, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        print(f"  [Embedder] Loading {model_name} on {device}...")
        self._model = SentenceTransformer(model_name, device=device)
        self._model.eval()
        self.dim = self._model.get_embedding_dimension()
        print(f"  [Embedder] Ready. dim={self.dim}")

    def embed(self, text: str) -> torch.Tensor:
        """
        Embed a single string. Returns a normalised (768,) float32 tensor.
        Equivalent to ham_brain.embed() but LLM-free.
        """
        with torch.no_grad():
            vec = self._model.encode(
                text,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=self.device,
            )
        return vec.float().to(self.device)

    def embed_batch(self, texts: list, batch_size: int = 128) -> torch.Tensor:
        """
        Embed a list of strings efficiently in batches.
        Returns a (N, dim) float32 tensor, each row L2-normalised.
        """
        with torch.no_grad():
            vecs = self._model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=self.device,
                show_progress_bar=len(texts) > 200,
            )
        return vecs.float().to(self.device)

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity between two embedding vectors."""
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ---------------------------------------------------------------------------
# Module-level singleton — mirrors the ham_brain.embed() interface
# ---------------------------------------------------------------------------

_default_embedder: Embedder | None = None


def get_embedder(model_name=DEFAULT_MODEL) -> Embedder:
    """Return the module-level singleton embedder, creating it if needed."""
    global _default_embedder
    if _default_embedder is None or _default_embedder.model_name != model_name:
        _default_embedder = Embedder(model_name)
    return _default_embedder


def embed(text: str) -> torch.Tensor:
    """
    Drop-in replacement for ham_brain.embed().
    Lazily loads the default embedder on first call.
    """
    return get_embedder().embed(text)


def embed_batch(texts: list, batch_size: int = 128) -> torch.Tensor:
    """Batch version of embed()."""
    return get_embedder().embed_batch(texts, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nSelf-test: ham_embedder")
    emb = Embedder()
    print(f"  dim = {emb.dim}")

    pairs = [
        ("number theory", "prime numbers"),
        ("number theory", "plate tectonics"),
        ("Pythagorean theorem", "right triangle"),
        ("Pythagorean theorem", "French cuisine"),
    ]

    print("\n  Similarity checks:")
    for a, b in pairs:
        sim = emb.similarity(emb.embed(a), emb.embed(b))
        print(f"    {a!r:35s} <-> {b!r:25s}  sim={sim:.4f}")

    batch = emb.embed_batch(["axiom", "theorem", "conjecture", "proof"])
    print(f"\n  Batch embed shape: {batch.shape}")
    print("  Self-test passed.")
