"""Tests for the EmbeddingEngine (RED phase).

EmbeddingEngine is the public interface for the AI memory system's
text-vectorisation layer.  These tests exercise only the public API
— encode, similarity, encode_batch — so the implementation behind
the interface is free to change.
"""

from __future__ import annotations

import math

import pytest

from graph_memory.embeddings import EmbeddingEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> EmbeddingEngine:
    """Return an EmbeddingEngine loaded with the small multilingual model."""
    return EmbeddingEngine(model_name="paraphrase-multilingual-MiniLM-L12-v2")


# ---------------------------------------------------------------------------
# Construction & model loading
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestEmbeddingEngine:
    """EmbeddingEngine lifecycle and core operations."""

    def test_init_loads_model(self) -> None:
        """EmbeddingEngine() constructs successfully and loads a model."""
        eng = EmbeddingEngine(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        assert eng is not None
        # The underlying model must be accessible.
        assert hasattr(eng, "model")

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------

    def test_encode_returns_floats(self, engine: EmbeddingEngine) -> None:
        """encode(text) returns a list of Python floats."""
        vec = engine.encode("Python uses indentation for code blocks")
        assert isinstance(vec, list), "encode must return a list"
        assert len(vec) > 0, "encode must return a non-empty list"
        assert all(
            isinstance(v, float) for v in vec
        ), "all elements must be floats"

    def test_encode_consistent_dimension(self, engine: EmbeddingEngine) -> None:
        """encode(text) always returns a vector of the same dimension."""
        texts = [
            "short",
            "A longer sentence with several words in it.",
            "Python uses indentation for code blocks",
        ]
        dims = {len(engine.encode(t)) for t in texts}
        assert len(dims) == 1, (
            f"encode returned inconsistent dimensions: {dims}"
        )

    # ------------------------------------------------------------------
    # similarity
    # ------------------------------------------------------------------

    def test_similarity_same_text(self, engine: EmbeddingEngine) -> None:
        """similarity(encode(a), encode(a)) is approximately 1.0."""
        text = "Python uses indentation for code blocks"
        vec = engine.encode(text)
        sim = engine.similarity(vec, vec)
        assert isinstance(sim, float), "similarity must return a float"
        assert math.isclose(sim, 1.0, abs_tol=1e-4), (
            f"same-text similarity expected ~1.0, got {sim}"
        )

    def test_similarity_related_texts(self, engine: EmbeddingEngine) -> None:
        """Related texts have similarity > 0.5."""
        a = engine.encode("Python is a programming language")
        b = engine.encode("Python is used for coding")
        sim = engine.similarity(a, b)
        assert sim > 0.5, (
            f"related-text similarity expected >0.5, got {sim}"
        )

    def test_similarity_unrelated_texts(self, engine: EmbeddingEngine) -> None:
        """Unrelated texts have similarity < 0.5."""
        a = engine.encode("Python programming")
        b = engine.encode("The weather is sunny")
        sim = engine.similarity(a, b)
        assert sim < 0.5, (
            f"unrelated-text similarity expected <0.5, got {sim}"
        )

    # ------------------------------------------------------------------
    # batch encode
    # ------------------------------------------------------------------

    def test_batch_encode(self, engine: EmbeddingEngine) -> None:
        """encode_batch returns one vector per input text."""
        texts = [
            "Python uses indentation for code blocks",
            "JavaScript runs in the browser",
            "Rust is a systems programming language",
        ]
        vectors = engine.encode_batch(texts)
        assert isinstance(vectors, list), "encode_batch must return a list"
        assert len(vectors) == len(texts), (
            f"encode_batch returned {len(vectors)} vectors, "
            f"expected {len(texts)}"
        )
        for vec in vectors:
            assert isinstance(vec, list), "each element must be a list"
            assert len(vec) > 0, "each vector must be non-empty"
            assert all(
                isinstance(v, float) for v in vec
            ), "all elements must be floats"

        # All vectors must share the same dimension.
        dims = {len(v) for v in vectors}
        assert len(dims) == 1, (
            f"encode_batch returned inconsistent dimensions: {dims}"
        )
