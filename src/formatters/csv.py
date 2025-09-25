"""
CSV formatter
"""

import csv
from typing import Any

class CSVFormatter:
    """Export dataset to CSV format"""

    def format(self, dataset: Any, path: str, **kwargs):
        """Format dataset as CSV"""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['id', 'text', 'source', 'timestamp', 'quality_score', 'language']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for doc in dataset.documents:
                row = {
                    'id': doc.id,
                    'text': doc.text[:1000],  # Truncate for CSV
                    'source': doc.source,
                    'timestamp': doc.timestamp.isoformat(),
                    'quality_score': getattr(doc, 'quality_score', ''),
                    'language': getattr(doc, 'language', '')
                }
                writer.writerow(row)