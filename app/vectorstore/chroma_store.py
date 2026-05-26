"""
ChromaDB vector store wrapper.
Handles document storage, embedding indexing, and similarity search.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from app.models.schemas import RetrievedChunk
from app.utils.logger import logger


class ChromaStore:
    """
    Wrapper around ChromaDB for persistent vector storage.
    Uses cosine similarity for retrieval.
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        """
        Initialize ChromaDB with persistent storage.
        
        Args:
            persist_dir: Directory for persistent vector storage.
        """
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hf:space": "knowledge_base", "hf:distance": "cosine"},
        )
        logger.info(
            f"ChromaDB initialized at '{persist_dir}' "
            f"(collection: knowledge_base, "
            f"existing docs: {self._collection.count()})"
        )

    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """
        Add document chunks with their embeddings to the vector store.
        
        Args:
            chunks: List of chunk dicts with 'text' and 'metadata' keys.
            embeddings: Corresponding embedding vectors.
        """
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings to add")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = chunk["metadata"].get("chunk_id", f"chunk_{i}")
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append({
                "title": chunk["metadata"].get("title", "Unknown"),
                "chunk_id": chunk_id,
                "source": chunk["metadata"].get("source", "Unknown"),
                "chunk_index": str(chunk["metadata"].get("chunk_index", 0)),
            })

        # Upsert to handle re-indexing without duplicates
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(
            f"Upserted {len(chunks)} chunks into ChromaDB "
            f"(total: {self._collection.count()})"
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        similarity_threshold: float = 0.35,
    ) -> List[RetrievedChunk]:
        """
        Perform similarity search against stored embeddings.
        
        ChromaDB uses cosine distance internally (distance = 1 - similarity).
        We convert back to cosine similarity for threshold comparison.
        
        Args:
            query_embedding: The query vector.
            top_k: Number of top results to retrieve.
            similarity_threshold: Minimum similarity score to include.
            
        Returns:
            List of RetrievedChunk objects sorted by similarity (descending).
        """
        if self._collection.count() == 0:
            logger.warning("Vector store is empty, no results to return")
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieved_chunks = []

        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # Convert cosine distance to cosine similarity
                distance = results["distances"][0][i]
                similarity = 1 - distance

                metadata = results["metadatas"][0][i]

                # Log similarity scores for debugging
                logger.info(
                    f"  Chunk '{metadata.get('chunk_id', 'unknown')}' | "
                    f"Similarity: {similarity:.4f} | "
                    f"Source: {metadata.get('source', 'unknown')}"
                )

                # Apply similarity threshold
                if similarity >= similarity_threshold:
                    retrieved_chunks.append(
                        RetrievedChunk(
                            text=doc,
                            title=metadata.get("title", "Unknown"),
                            chunk_id=metadata.get("chunk_id", f"chunk_{i}"),
                            source=metadata.get("source", "Unknown"),
                            similarity_score=round(similarity, 4),
                        )
                    )

        # Sort by similarity descending
        retrieved_chunks.sort(
            key=lambda x: x.similarity_score, reverse=True
        )

        logger.info(
            f"Search returned {len(retrieved_chunks)} chunks "
            f"above threshold ({similarity_threshold})"
        )

        return retrieved_chunks

    @property
    def count(self) -> int:
        """Return total number of stored chunks."""
        try:
            return self._collection.count()
        except Exception as e:
            logger.warning(f"Error counting chunks: {e}")
            return 0

    def reset(self) -> None:
        """Clear all documents from the collection."""
        self._client.delete_collection("knowledge_base")
        self._collection = self._client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hf:space": "knowledge_base", "hf:distance": "cosine"},
        )
        logger.info("ChromaDB collection reset")
