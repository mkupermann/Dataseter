"""
HuggingFace datasets formatter
"""

from datasets import Dataset, DatasetDict
from typing import Any

class HuggingFaceFormatter:
    """Export to HuggingFace datasets format"""

    def format(self, dataset: Any, name: str, **kwargs):
        """Format dataset for HuggingFace"""
        records = {
            'id': [],
            'text': [],
            'source': [],
            'metadata': []
        }
        
        for doc in dataset.documents:
            records['id'].append(doc.id)
            records['text'].append(doc.text)
            records['source'].append(doc.source)
            records['metadata'].append(doc.metadata)
        
        hf_dataset = Dataset.from_dict(records)
        
        if kwargs.get('push_to_hub', False):
            hf_dataset.push_to_hub(name)
        else:
            hf_dataset.save_to_disk(name)