"""
Main dataset analyzer
"""

from typing import Dict, Any
from collections import Counter
import statistics

class DatasetAnalyzer:
    """Comprehensive dataset analysis"""

    def analyze(self, dataset: Any) -> Dict[str, Any]:
        """Analyze dataset and return statistics"""
        stats = {
            'total_documents': len(dataset.documents),
            'total_chunks': sum(len(doc.chunks) if hasattr(doc, 'chunks') else 0
                              for doc in dataset.documents),
            'total_text_length': sum(len(doc.text) for doc in dataset.documents),
            'sources': Counter(doc.source for doc in dataset.documents),
            'languages': Counter(getattr(doc, 'language', 'unknown')
                               for doc in dataset.documents),
        }

        # Calculate quality statistics
        quality_scores = [doc.quality_score for doc in dataset.documents
                         if hasattr(doc, 'quality_score')]
        if quality_scores:
            stats['quality_stats'] = {
                'mean': statistics.mean(quality_scores),
                'median': statistics.median(quality_scores),
                'min': min(quality_scores),
                'max': max(quality_scores),
                'stdev': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
            }

        # Calculate text length statistics
        text_lengths = [len(doc.text) for doc in dataset.documents]
        if text_lengths:
            stats['text_length_stats'] = {
                'mean': statistics.mean(text_lengths),
                'median': statistics.median(text_lengths),
                'min': min(text_lengths),
                'max': max(text_lengths),
                'total': sum(text_lengths)
            }
        else:
            stats['text_length_stats'] = {
                'mean': 0,
                'median': 0,
                'min': 0,
                'max': 0,
                'total': 0
            }

        # Word count statistics
        word_counts = [len(doc.text.split()) for doc in dataset.documents]
        if word_counts:
            stats['word_count_stats'] = {
                'mean': statistics.mean(word_counts),
                'median': statistics.median(word_counts),
                'min': min(word_counts),
                'max': max(word_counts),
                'total': sum(word_counts)
            }
        else:
            stats['word_count_stats'] = {
                'mean': 0,
                'median': 0,
                'min': 0,
                'max': 0,
                'total': 0
            }

        # Vocabulary analysis
        all_words = []
        for doc in dataset.documents:
            all_words.extend(doc.text.lower().split())
        stats['vocabulary_size'] = len(set(all_words))
        stats['total_words'] = len(all_words)

        return stats