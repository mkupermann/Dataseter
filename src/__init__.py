"""
Dataseter - Advanced AI Training Dataset Creator
"""

__version__ = "1.0.0"
__author__ = "Dataseter Team"
__email__ = "dataseter@example.com"

from .core import DatasetCreator, Pipeline
from .extractors import (
    PDFExtractor,
    WebExtractor,
    OfficeExtractor,
    EbookExtractor,
)
from .processors import (
    Chunker,
    Cleaner,
    Deduplicator,
    PrivacyProtector,
    QualityFilter,
)
from .formatters import (
    JSONLFormatter,
    ParquetFormatter,
    HuggingFaceFormatter,
    CSVFormatter,
)
from .analyzers import (
    DatasetAnalyzer,
    QualityAnalyzer,
    StatisticsCalculator,
    Visualizer,
)

__all__ = [
    "DatasetCreator",
    "Pipeline",
    "PDFExtractor",
    "WebExtractor",
    "OfficeExtractor",
    "EbookExtractor",
    "Chunker",
    "Cleaner",
    "Deduplicator",
    "PrivacyProtector",
    "QualityFilter",
    "JSONLFormatter",
    "ParquetFormatter",
    "HuggingFaceFormatter",
    "CSVFormatter",
    "DatasetAnalyzer",
    "QualityAnalyzer",
    "StatisticsCalculator",
    "Visualizer",
]