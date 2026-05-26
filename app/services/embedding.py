"""
Embedding service using Google Generative AI.
Generates vector embeddings for text using Google's embedding model.
"""

import google.generativeai as genai
from typing import List
import time
from app.config import get_settings
from app.utils.logger import logger


class EmbeddingService:
    """
    Generates text embeddings using Google's Generative AI embedding model.
    Handles batch processing, error recovery, and rate limiting.
    """

    def __init__(self):
        settings = get_settings()
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = "models/gemini-embedding-001"
        logger.info(f"Embedding service initialized with model: {self.model}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text to embed.
            
        Returns:
            List of floats representing the embedding vector.
            
        Raises:
            Exception: If embedding generation fails after retries.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document",
                )
                embedding = result["embedding"]
                logger.debug(
                    f"Generated embedding: {len(embedding)} dimensions"
                )
                return embedding

            except Exception as e:
                error_msg = str(e).lower()

                if "rate" in error_msg or "quota" in error_msg:
                    wait_time = (attempt + 1) * 5
                    logger.warning(
                        f"Rate limit hit, waiting {wait_time}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                elif "api key" in error_msg or "invalid" in error_msg:
                    logger.error("Invalid API key for embedding service")
                    raise ValueError(
                        "Invalid Gemini API key. Please check your "
                        "GEMINI_API_KEY in .env file."
                    )
                else:
                    logger.error(
                        f"Embedding error (attempt {attempt + 1}): {e}"
                    )
                    if attempt == max_retries - 1:
                        raise

                    time.sleep(2)

        raise RuntimeError("Failed to generate embedding after all retries")

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query text.
        Uses 'retrieval_query' task type for better search results.
        
        Args:
            text: Query text to embed.
            
        Returns:
            List of floats representing the query embedding.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_query",
                )
                return result["embedding"]

            except Exception as e:
                error_msg = str(e).lower()

                if "rate" in error_msg or "quota" in error_msg:
                    wait_time = (attempt + 1) * 5
                    logger.warning(
                        f"Rate limit on query embed, waiting {wait_time}s"
                    )
                    time.sleep(wait_time)
                elif "api key" in error_msg or "invalid" in error_msg:
                    raise ValueError(
                        "Invalid Gemini API key. Check GEMINI_API_KEY."
                    )
                else:
                    logger.error(f"Query embed error: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2)

        raise RuntimeError("Failed to generate query embedding")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        Processes texts individually with rate limiting to avoid quota issues.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors.
        """
        embeddings = []
        total = len(texts)

        logger.info(f"Generating embeddings for {total} texts...")

        for i, text in enumerate(texts):
            embedding = self.embed_text(text)
            embeddings.append(embedding)

            # Progress logging every 10 chunks
            if (i + 1) % 10 == 0 or (i + 1) == total:
                logger.info(
                    f"Embedding progress: {i + 1}/{total} "
                    f"({((i + 1) / total * 100):.0f}%)"
                )

            # Small delay to avoid rate limiting
            if i < total - 1:
                time.sleep(0.1)

        logger.info(f"Completed embedding generation for {total} texts")
        return embeddings
