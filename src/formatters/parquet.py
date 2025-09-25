"""
Parquet formatter
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Any

class ParquetFormatter:
    """Export dataset to Parquet format"""

    def format(self, dataset: Any, path: str, **kwargs):
        """Format dataset as Parquet"""
        records = []
        
        for doc in dataset.documents:
            record = {
                'id': doc.id,
                'text': doc.text,
                'source': doc.source,
                'timestamp': doc.timestamp,
                'metadata': str(doc.metadata)
            }
            
            if hasattr(doc, 'quality_score'):
                record['quality_score'] = doc.quality_score
            if hasattr(doc, 'language'):
                record['language'] = doc.language
                
            records.append(record)
        
        df = pd.DataFrame(records)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path, compression='snappy')