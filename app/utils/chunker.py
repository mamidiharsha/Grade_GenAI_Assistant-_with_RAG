"""
Document chunking utility.
Splits documents into smaller chunks suitable for embedding generation.
Uses sentence-aware splitting to avoid breaking mid-sentence.
"""

import re
from typing import List, Dict, Any
from app.utils.logger import logger


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.
    Uses a simple heuristic: ~1 token per 0.75 words (common for English text).
    
    Args:
        text: Input text string.
        
    Returns:
        Estimated token count.
    """
    words = len(text.split())
    return int(words / 0.75)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex-based sentence boundary detection.
    
    Args:
        text: Input text to split.
        
    Returns:
        List of sentences.
    """
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(
    document: Dict[str, str],
    max_tokens: int = 400,
    overlap_tokens: int = 50,
) -> List[Dict[str, Any]]:
    """
    Split a document into chunks with metadata.
    
    Uses sentence-aware splitting to ensure chunks don't break mid-sentence.
    Includes overlap between consecutive chunks to preserve context.
    
    Args:
        document: Dict with 'title' and 'content' keys.
        max_tokens: Maximum tokens per chunk (default: 400).
        overlap_tokens: Number of overlapping tokens between chunks (default: 50).
        
    Returns:
        List of chunk dictionaries with 'text', 'metadata' keys.
    """
    title = document.get("title", "Untitled")
    content = document.get("content", "")

    if not content.strip():
        logger.warning(f"Empty content for document: {title}")
        return []

    sentences = split_into_sentences(content)
    chunks = []
    current_chunk_sentences = []
    current_token_count = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        # If adding this sentence would exceed max tokens, finalize current chunk
        if current_token_count + sentence_tokens > max_tokens and current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "title": title,
                    "chunk_id": f"{title.lower().replace(' ', '_')}_{chunk_index}",
                    "source": title,
                    "chunk_index": chunk_index,
                },
            })
            chunk_index += 1

            # Calculate overlap: keep last few sentences that fit within overlap_tokens
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_chunk_sentences):
                s_tokens = estimate_tokens(s)
                if overlap_count + s_tokens <= overlap_tokens:
                    overlap_sentences.insert(0, s)
                    overlap_count += s_tokens
                else:
                    break

            current_chunk_sentences = overlap_sentences
            current_token_count = overlap_count

        current_chunk_sentences.append(sentence)
        current_token_count += sentence_tokens

    # Add the final chunk
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "title": title,
                "chunk_id": f"{title.lower().replace(' ', '_')}_{chunk_index}",
                "source": title,
                "chunk_index": chunk_index,
            },
        })

    logger.info(
        f"Document '{title}' chunked into {len(chunks)} chunks "
        f"(max_tokens={max_tokens}, overlap={overlap_tokens})"
    )

    return chunks


def chunk_all_documents(
    documents: List[Dict[str, str]],
    max_tokens: int = 400,
    overlap_tokens: int = 50,
) -> List[Dict[str, Any]]:
    """
    Chunk all documents in the knowledge base.
    
    Args:
        documents: List of document dicts with 'title' and 'content'.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Overlap tokens between chunks.
        
    Returns:
        List of all chunks across all documents.
    """
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, max_tokens, overlap_tokens)
        all_chunks.extend(chunks)

    logger.info(
        f"Total: {len(all_chunks)} chunks from {len(documents)} documents"
    )
    return all_chunks
