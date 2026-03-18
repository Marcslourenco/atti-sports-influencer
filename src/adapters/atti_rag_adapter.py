"""
ATTI RAG Adapter — Real integration with FAISS 384d vector indices.
Connects to: knowledge/vector_indices/atti_tax_knowledge_faiss.bin
             knowledge/vector_indices/atti_embeddings.npy
             knowledge/rag/ATTI_RAG_KNOWLEDGE_MATRIX_v2.json

Features:
- Direct FAISS index loading (binary, not JSON placeholder)
- Sentence-transformers embedding generation
- Knowledge Package loading (ATTI JSON format)
- Hybrid search: vector similarity + keyword scoring
- Context builder compatible with ATTI ContextBuilder pattern
- Sports-specific knowledge injection at runtime
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

logger = logging.getLogger(__name__)


class ATTIRAGAdapter:
    """
    Real adapter for ATTI's FAISS-based RAG infrastructure.

    Loads:
    1. FAISS binary index (atti_tax_knowledge_faiss.bin)
    2. Embeddings numpy array (atti_embeddings.npy)
    3. Knowledge matrices (ATTI_RAG_KNOWLEDGE_MATRIX_v2.json)
    4. Sports-specific knowledge packages (injected at runtime)

    Search modes:
    - vector: FAISS similarity search (requires sentence-transformers)
    - keyword: TF-IDF-like keyword matching
    - hybrid: vector + keyword with weighted scoring
    """

    def __init__(
        self,
        knowledge_base_dir: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        search_mode: str = "hybrid",
    ):
        self.knowledge_base_dir = Path(
            knowledge_base_dir
            or os.getenv(
                "ATTI_KNOWLEDGE_DIR",
                "/home/ubuntu/atti-agent-template/knowledge"
            )
        )
        self.embedding_model_name = embedding_model
        self.embedding_dim = embedding_dim
        self.search_mode = search_mode

        self._index = None
        self._embeddings = None
        self._chunks: List[Dict[str, Any]] = []
        self._sports_chunks: List[Dict[str, Any]] = []
        self._encoder = None
        self._initialized = False

    def initialize(self) -> bool:
        """Load FAISS index, embeddings, and knowledge matrices"""
        try:
            self._load_knowledge_matrices()
            self._load_faiss_index()
            self._load_embeddings()
            self._initialized = True
            logger.info(
                f"RAG adapter initialized: {len(self._chunks)} chunks, "
                f"FAISS={self._index is not None}, "
                f"embeddings={self._embeddings is not None}"
            )
            return True
        except Exception as e:
            logger.error(f"RAG initialization failed: {e}")
            self._initialized = True  # Allow keyword-only search
            return False

    def _load_knowledge_matrices(self):
        """Load ATTI RAG Knowledge Matrices (JSON format)"""
        rag_dir = self.knowledge_base_dir / "rag"
        if not rag_dir.exists():
            logger.warning(f"RAG directory not found: {rag_dir}")
            return

        for json_file in rag_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # ATTI format: list of knowledge blocks
                if isinstance(data, list):
                    for block in data:
                        chunk = {
                            "id": block.get("id", f"chunk_{len(self._chunks)}"),
                            "content": block.get("conteudo", block.get("content", block.get("text", ""))),
                            "category": block.get("categoria_macro", block.get("category", "general")),
                            "subcategory": block.get("subcategoria", ""),
                            "tags": block.get("tags", []),
                            "source": str(json_file.name),
                            "priority": block.get("prioridade_contextual", block.get("priority", 0.5)),
                            "package": block.get("package", json_file.stem),
                        }
                        if chunk["content"]:
                            self._chunks.append(chunk)

                # ATTI format: dict with knowledge_blocks or package_metadata
                elif isinstance(data, dict):
                    blocks = data.get("knowledge_blocks", data.get("blocks", []))
                    metadata = data.get("package_metadata", {})
                    for block in blocks:
                        chunk = {
                            "id": block.get("id", f"chunk_{len(self._chunks)}"),
                            "content": block.get("conteudo", block.get("content", block.get("text", ""))),
                            "category": block.get("categoria_macro", block.get("category", metadata.get("domain", "general"))),
                            "subcategory": block.get("subcategoria", ""),
                            "tags": block.get("tags", []),
                            "source": str(json_file.name),
                            "priority": block.get("prioridade_contextual", block.get("priority", 0.5)),
                            "package": metadata.get("segmento", metadata.get("name", json_file.stem)),
                        }
                        if chunk["content"]:
                            self._chunks.append(chunk)

                logger.info(f"Loaded {json_file.name}: {len(self._chunks)} total chunks")
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")

        # Also load reference knowledge
        ref_dir = self.knowledge_base_dir / "reference"
        if ref_dir.exists():
            for json_file in ref_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    blocks = []
                    if isinstance(data, list):
                        blocks = data
                    elif isinstance(data, dict):
                        blocks = data.get("knowledge_blocks", data.get("entries", []))
                    for block in blocks:
                        content = block.get("conteudo", block.get("content", block.get("text", "")))
                        if content:
                            self._chunks.append({
                                "id": block.get("id", f"ref_{len(self._chunks)}"),
                                "content": content,
                                "category": "reference",
                                "tags": block.get("tags", []),
                                "source": str(json_file.name),
                                "priority": 0.3,
                                "package": json_file.stem,
                            })
                    logger.info(f"Loaded reference: {json_file.name}")
                except Exception as e:
                    logger.error(f"Error loading reference {json_file}: {e}")

    def _load_faiss_index(self):
        """Load FAISS binary index"""
        index_dir = self.knowledge_base_dir / "vector_indices"
        faiss_files = list(index_dir.glob("*faiss*.bin")) if index_dir.exists() else []

        if not faiss_files:
            logger.warning("No FAISS index files found")
            return

        try:
            import faiss
            for faiss_file in faiss_files:
                file_size = faiss_file.stat().st_size
                if file_size < 500:
                    logger.warning(f"FAISS file too small ({file_size}B), likely placeholder: {faiss_file}")
                    continue

                self._index = faiss.read_index(str(faiss_file))
                logger.info(
                    f"FAISS index loaded: {faiss_file.name}, "
                    f"{self._index.ntotal} vectors, dim={self._index.d}"
                )
                break
        except ImportError:
            logger.warning("faiss-cpu not installed, vector search disabled")
        except Exception as e:
            logger.error(f"FAISS load error: {e}")

    def _load_embeddings(self):
        """Load pre-computed embeddings"""
        index_dir = self.knowledge_base_dir / "vector_indices"
        npy_files = list(index_dir.glob("*embeddings*.npy")) if index_dir.exists() else []

        if not npy_files:
            logger.warning("No embedding files found")
            return

        try:
            self._embeddings = np.load(str(npy_files[0]))
            logger.info(
                f"Embeddings loaded: {npy_files[0].name}, "
                f"shape={self._embeddings.shape}"
            )
        except Exception as e:
            logger.error(f"Embeddings load error: {e}")

    def inject_sports_knowledge(self, knowledge_data: List[Dict[str, Any]]):
        """Inject sports-specific knowledge at runtime (live data, stats, etc.)"""
        for item in knowledge_data:
            chunk = {
                "id": item.get("id", f"sports_{len(self._sports_chunks)}"),
                "content": item.get("content", ""),
                "category": item.get("category", "sports"),
                "tags": item.get("tags", ["sports"]),
                "source": "sports_injection",
                "priority": item.get("priority", 0.7),
                "package": "sports_live",
            }
            if chunk["content"]:
                self._sports_chunks.append(chunk)

        logger.info(f"Injected {len(knowledge_data)} sports knowledge chunks")

    def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        include_sports: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base with hybrid approach.

        Args:
            query: Search query
            top_k: Number of results
            category_filter: Optional category filter
            include_sports: Include sports-specific chunks

        Returns:
            List of matching chunks with scores
        """
        all_chunks = list(self._chunks)
        if include_sports:
            all_chunks.extend(self._sports_chunks)

        results = []

        # Vector search (if FAISS available)
        if self.search_mode in ("vector", "hybrid") and self._index and self._embeddings is not None:
            vector_results = self._vector_search(query, top_k * 2)
            results.extend(vector_results)

        # Keyword search
        if self.search_mode in ("keyword", "hybrid"):
            keyword_results = self._keyword_search(query, top_k * 2, all_chunks, category_filter)
            results.extend(keyword_results)

        # Deduplicate and merge scores
        seen = {}
        for r in results:
            rid = r.get("id", id(r))
            if rid in seen:
                seen[rid]["search_score"] = max(
                    seen[rid].get("search_score", 0),
                    r.get("search_score", 0)
                )
            else:
                seen[rid] = r

        # Sort by combined score + priority
        merged = list(seen.values())
        merged.sort(
            key=lambda x: x.get("search_score", 0) * 0.7 + x.get("priority", 0.5) * 0.3,
            reverse=True,
        )
        return merged[:top_k]

    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """FAISS vector similarity search"""
        try:
            query_embedding = self._encode_query(query)
            if query_embedding is None:
                return []

            import faiss
            query_vec = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vec)

            distances, indices = self._index.search(query_vec, min(top_k, self._index.ntotal))

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self._chunks):
                    continue
                chunk = dict(self._chunks[idx])
                chunk["search_score"] = float(1.0 / (1.0 + dist))
                chunk["search_method"] = "vector"
                results.append(chunk)

            return results
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    def _encode_query(self, query: str) -> Optional[np.ndarray]:
        """Encode query using sentence-transformers"""
        try:
            if self._encoder is None:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.embedding_model_name)
            embedding = self._encoder.encode(query, normalize_embeddings=True)
            return embedding
        except ImportError:
            logger.warning("sentence-transformers not installed")
            return None
        except Exception as e:
            logger.error(f"Encoding error: {e}")
            return None

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        chunks: List[Dict],
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Keyword search with TF-IDF-like scoring"""
        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        scored = []

        for chunk in chunks:
            if category_filter and chunk.get("category") != category_filter:
                continue

            content = chunk.get("content", "").lower()
            tags = [t.lower() for t in chunk.get("tags", [])]

            score = 0
            for term in query_terms:
                if term in content:
                    score += content.count(term) * 1.0
                if any(term in tag for tag in tags):
                    score += 3.0

            if score > 0:
                result = dict(chunk)
                result["search_score"] = score
                result["search_method"] = "keyword"
                scored.append(result)

        scored.sort(key=lambda x: x["search_score"], reverse=True)
        return scored[:top_k]

    def build_context(
        self, query: str, max_chunks: int = 5, include_sports: bool = True
    ) -> Dict[str, Any]:
        """
        Build context string from search results.
        Compatible with ATTI ContextBuilder pattern.
        """
        results = self.search(query, top_k=max_chunks, include_sports=include_sports)

        if not results:
            return {"context": "", "sources": [], "num_sources": 0}

        context_parts = []
        sources = []
        for i, r in enumerate(results, 1):
            context_parts.append(f"[{i}] {r.get('content', '')}")
            sources.append({
                "id": r.get("id", ""),
                "category": r.get("category", ""),
                "package": r.get("package", ""),
                "score": r.get("search_score", 0),
                "method": r.get("search_method", "unknown"),
            })

        return {
            "context": "\n\n".join(context_parts),
            "sources": sources,
            "num_sources": len(sources),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "total_chunks": len(self._chunks),
            "sports_chunks": len(self._sports_chunks),
            "faiss_loaded": self._index is not None,
            "faiss_vectors": self._index.ntotal if self._index else 0,
            "embeddings_loaded": self._embeddings is not None,
            "embeddings_shape": list(self._embeddings.shape) if self._embeddings is not None else None,
            "embedding_dim": self.embedding_dim,
            "search_mode": self.search_mode,
            "knowledge_dir": str(self.knowledge_base_dir),
        }
