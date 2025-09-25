"""
Text chunking module with multiple strategies
"""

import re
from typing import List, Dict, Any, Optional
import logging

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


class Chunker:
    """Text chunking with various strategies"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.strategy = config.get('strategy', 'sliding_window')
        self.chunk_size = config.get('size', 512)
        self.overlap = config.get('overlap', 50)
        self.min_chunk_size = config.get('min_chunk_size', 100)
        self.max_chunk_size = config.get('max_chunk_size', 2048)
        self.preserve_sentences = config.get('preserve_sentences', True)

    def process(self, document: Any, **kwargs) -> Any:
        """Process document and create chunks"""
        if hasattr(document, 'text'):
            text = document.text
            chunks = self.chunk_text(text, **kwargs)
            document.chunks = chunks
        return document

    def chunk_text(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Chunk text using configured strategy"""
        strategy = kwargs.get('strategy', self.strategy)

        if strategy == 'fixed':
            return self._fixed_chunking(text, **kwargs)
        elif strategy == 'sliding_window':
            return self._sliding_window_chunking(text, **kwargs)
        elif strategy == 'sentence':
            return self._sentence_chunking(text, **kwargs)
        elif strategy == 'paragraph':
            return self._paragraph_chunking(text, **kwargs)
        elif strategy == 'semantic':
            return self._semantic_chunking(text, **kwargs)
        else:
            logger.warning(f"Unknown strategy: {strategy}, using sliding_window")
            return self._sliding_window_chunking(text, **kwargs)

    def _fixed_chunking(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Fixed size chunks without overlap"""
        chunk_size = kwargs.get('size', self.chunk_size)
        chunks = []

        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)

            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'start_index': i,
                    'end_index': min(i + chunk_size, len(words)),
                    'word_count': len(chunk_words)
                })

        return chunks

    def _sliding_window_chunking(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Sliding window chunks with overlap"""
        chunk_size = kwargs.get('size', self.chunk_size)
        overlap = kwargs.get('overlap', self.overlap)
        chunks = []

        words = text.split()
        step = chunk_size - overlap

        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)

            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'start_index': i,
                    'end_index': min(i + chunk_size, len(words)),
                    'word_count': len(chunk_words),
                    'overlap_prev': overlap if i > 0 else 0,
                    'overlap_next': overlap if i + chunk_size < len(words) else 0
                })

            if i + chunk_size >= len(words):
                break

        return chunks

    def _sentence_chunking(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Chunk by sentences"""
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
            except:
                nltk.download('punkt', quiet=True)
                sentences = sent_tokenize(text)
        else:
            sentences = re.split(r'[.!?]+', text)

        chunk_size = kwargs.get('size', self.chunk_size)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_size = len(sentence_words)

            if current_size + sentence_size > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'sentence_count': len(current_chunk),
                    'word_count': current_size
                })
                current_chunk = []
                current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'sentence_count': len(current_chunk),
                    'word_count': current_size
                })

        return chunks

    def _paragraph_chunking(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Chunk by paragraphs"""
        paragraphs = re.split(r'\n\n+', text)
        chunk_size = kwargs.get('size', self.chunk_size)
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_words = para.split()
            para_size = len(para_words)

            if current_size + para_size > chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'paragraph_count': len(current_chunk),
                    'word_count': current_size
                })
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_size

        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'paragraph_count': len(current_chunk),
                    'word_count': current_size
                })

        return chunks

    def _semantic_chunking(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Semantic chunking based on content structure"""
        # This is a simplified version
        # A real implementation would use embeddings or NLP models

        # Split by headers or natural breaks
        sections = re.split(r'\n(?=[A-Z#*-])', text)
        chunk_size = kwargs.get('size', self.chunk_size)
        chunks = []

        for section in sections:
            section_words = section.split()

            if len(section_words) > chunk_size:
                # If section is too large, use sentence chunking
                sub_chunks = self._sentence_chunking(section, **kwargs)
                chunks.extend(sub_chunks)
            elif len(section) >= self.min_chunk_size:
                chunks.append({
                    'text': section,
                    'type': 'semantic_section',
                    'word_count': len(section_words)
                })

        return chunks