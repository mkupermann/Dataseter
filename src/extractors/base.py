"""
Base extractor class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import hashlib
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Abstract base class for all extractors"""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the extractor

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.stats = {
            'files_processed': 0,
            'bytes_processed': 0,
            'errors': 0,
            'start_time': datetime.now()
        }

    @abstractmethod
    def extract(self, source: str, **kwargs) -> Dict[str, Any]:
        """Extract text and metadata from source

        Args:
            source: Path to file or URL
            **kwargs: Additional extraction options

        Returns:
            Dictionary containing:
                - text: Extracted text content
                - metadata: Metadata about the source
                - error: Any error message (if applicable)
        """
        pass

    def validate_source(self, source: str) -> bool:
        """Validate that the source exists and is accessible

        Args:
            source: Path to file or URL

        Returns:
            True if source is valid, False otherwise
        """
        if source.startswith(('http://', 'https://')):
            # URL validation would go here
            return True
        else:
            path = Path(source)
            return path.exists() and path.is_file()

    def generate_id(self, content: str) -> str:
        """Generate a unique ID for the content

        Args:
            content: Text content

        Returns:
            Unique ID string
        """
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def extract_metadata(self, source: str) -> Dict[str, Any]:
        """Extract basic metadata from source

        Args:
            source: Path to file or URL

        Returns:
            Dictionary of metadata
        """
        metadata = {
            'source': source,
            'extraction_timestamp': datetime.now().isoformat(),
            'extractor': self.__class__.__name__
        }

        if not source.startswith(('http://', 'https://')):
            path = Path(source)
            if path.exists():
                stat = path.stat()
                metadata.update({
                    'file_size': stat.st_size,
                    'file_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'file_created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'file_name': path.name,
                    'file_extension': path.suffix
                })

        return metadata

    def update_stats(self, success: bool = True, bytes_processed: int = 0):
        """Update extraction statistics

        Args:
            success: Whether extraction was successful
            bytes_processed: Number of bytes processed
        """
        self.stats['files_processed'] += 1
        self.stats['bytes_processed'] += bytes_processed
        if not success:
            self.stats['errors'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics

        Returns:
            Dictionary of statistics
        """
        runtime = (datetime.now() - self.stats['start_time']).total_seconds()
        return {
            **self.stats,
            'runtime_seconds': runtime,
            'success_rate': (
                (self.stats['files_processed'] - self.stats['errors'])
                / max(self.stats['files_processed'], 1)
            ) * 100
        }

    def reset_stats(self):
        """Reset extraction statistics"""
        self.stats = {
            'files_processed': 0,
            'bytes_processed': 0,
            'errors': 0,
            'start_time': datetime.now()
        }