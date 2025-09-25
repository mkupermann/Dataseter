"""
Quality filtering and scoring module with advanced semantic quality assessment
"""

import re
import logging
from typing import Dict, Any
from langdetect import detect, LangDetectException

try:
    from .semantic_quality import SemanticQualityScorer
    SEMANTIC_QUALITY_AVAILABLE = True
except ImportError:
    SEMANTIC_QUALITY_AVAILABLE = False

logger = logging.getLogger(__name__)


class QualityFilter:
    """Filter content based on quality metrics"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_score = config.get('min_score', 0.7)
        self.detect_language = config.get('detect_language', True)
        self.allowed_languages = set(config.get('allowed_languages', ['en']))
        self.min_word_count = config.get('min_word_count', 10)
        self.max_word_count = config.get('max_word_count', 10000)
        self.min_avg_word_length = config.get('min_avg_word_length', 3)
        self.max_repetition_ratio = config.get('max_repetition_ratio', 0.3)
        self.use_semantic_scoring = config.get('use_semantic_scoring', True)

        # Initialize semantic quality scorer if available
        self.semantic_scorer = None
        if SEMANTIC_QUALITY_AVAILABLE and self.use_semantic_scoring:
            try:
                self.semantic_scorer = SemanticQualityScorer(config.get('semantic_quality', {}))
                logger.info("Semantic quality scoring enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic quality scorer: {e}")
                self.semantic_scorer = None

    def process(self, document: Any, **kwargs) -> Any:
        """Filter document based on quality"""
        threshold = kwargs.get('threshold', self.min_score)

        if hasattr(document, 'text'):
            # Use advanced semantic scoring if available
            if self.semantic_scorer:
                try:
                    # Process with semantic scorer (adds detailed quality analysis)
                    document = self.semantic_scorer.process(document, **kwargs)
                    score = document.quality_score if hasattr(document, 'quality_score') else 0.5
                except Exception as e:
                    logger.warning(f"Semantic quality scoring failed: {e}")
                    score = self.calculate_quality_score(document.text)
                    document.quality_score = score
            else:
                # Fallback to basic quality scoring
                score = self.calculate_quality_score(document.text)
                document.quality_score = score

            # Detect language
            if self.detect_language:
                try:
                    document.language = detect(document.text[:1000])
                except LangDetectException:
                    document.language = 'unknown'

            # Filter based on quality
            if score < threshold:
                logger.debug(f"Document filtered out with quality score {score:.2f}")
                return None

            # Filter based on language
            if self.detect_language and document.language not in self.allowed_languages:
                logger.debug(f"Document filtered out with language {document.language}")
                return None

        # Process chunks if present
        if hasattr(document, 'chunks'):
            filtered_chunks = []
            for chunk in document.chunks:
                if 'text' in chunk:
                    score = self.calculate_quality_score(chunk['text'])
                    chunk['quality_score'] = score
                    if score >= threshold:
                        filtered_chunks.append(chunk)
            document.chunks = filtered_chunks

        return document

    def calculate_quality_score(self, text: str) -> float:
        """Calculate quality score for text"""
        if not text:
            return 0.0

        scores = []

        # Word count score
        word_count = len(text.split())
        if word_count < self.min_word_count:
            scores.append(0.0)
        elif word_count > self.max_word_count:
            scores.append(0.5)
        else:
            scores.append(1.0)

        # Average word length score
        words = text.split()
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            if avg_word_length >= self.min_avg_word_length:
                scores.append(1.0)
            else:
                scores.append(avg_word_length / self.min_avg_word_length)

        # Repetition score
        repetition_ratio = self._calculate_repetition_ratio(text)
        if repetition_ratio <= self.max_repetition_ratio:
            scores.append(1.0)
        else:
            scores.append(1.0 - repetition_ratio)

        # Special character ratio
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / max(len(text), 1)
        scores.append(1.0 - min(special_char_ratio * 2, 1.0))

        # Uppercase ratio (too much uppercase is low quality)
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if uppercase_ratio > 0.5:
            scores.append(0.5)
        else:
            scores.append(1.0)

        # Calculate final score
        return sum(scores) / len(scores) if scores else 0.0

    def _calculate_repetition_ratio(self, text: str) -> float:
        """Calculate ratio of repeated content"""
        words = text.lower().split()
        if len(words) < 10:
            return 0.0

        # Check for repeated sequences
        sequence_length = 3
        repeated_count = 0

        for i in range(len(words) - sequence_length * 2):
            sequence = ' '.join(words[i:i + sequence_length])
            remaining = ' '.join(words[i + sequence_length:])
            if sequence in remaining:
                repeated_count += 1

        return repeated_count / max(len(words) - sequence_length, 1)