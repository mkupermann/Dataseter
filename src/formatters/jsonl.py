"""
JSONL (JSON Lines) formatter
"""

import json
import gzip
from pathlib import Path
from typing import Any

class JSONLFormatter:
    """Export dataset to JSONL format"""

    def format(self, dataset: Any, path: str, **kwargs):
        """Format dataset as JSONL"""
        compress = kwargs.get('compress', False)
        include_metadata = kwargs.get('include_metadata', True)

        open_func = gzip.open if compress else open
        mode = 'wt' if compress else 'w'

        with open_func(path, mode, encoding='utf-8') as f:
            for doc in dataset.documents:
                record = {
                    'id': doc.id,
                    'text': doc.text,
                    'source': doc.source,
                    'timestamp': doc.timestamp.isoformat()
                }

                if include_metadata:
                    record['metadata'] = doc.metadata

                if doc.chunks:
                    record['chunks'] = doc.chunks

                if hasattr(doc, 'quality_score'):
                    record['quality_score'] = doc.quality_score

                if hasattr(doc, 'language'):
                    record['language'] = doc.language

                f.write(json.dumps(record, ensure_ascii=False) + '\n')