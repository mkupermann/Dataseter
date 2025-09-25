"""
Core dataset creation functionality
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import yaml
from tqdm import tqdm

from .extractors import (
    PDFExtractor,
    WebExtractor,
    OfficeExtractor,
    EbookExtractor,
    TextExtractor,
)
from .processors import (
    Chunker,
    Cleaner,
    Deduplicator,
    PrivacyProtector,
    QualityFilter,
)
try:
    from .processors.knowledge_extractor import KnowledgeExtractor
    KNOWLEDGE_EXTRACTION_AVAILABLE = True
except ImportError:
    KNOWLEDGE_EXTRACTION_AVAILABLE = False

try:
    from .processors.metacognitive_annotator import MetacognitiveAnnotator
    METACOGNITIVE_ANNOTATION_AVAILABLE = True
except ImportError:
    METACOGNITIVE_ANNOTATION_AVAILABLE = False

try:
    from .processors.adversarial_tester import AdversarialTester
    ADVERSARIAL_TESTING_AVAILABLE = True
except ImportError:
    ADVERSARIAL_TESTING_AVAILABLE = False
from .formatters import (
    JSONLFormatter,
    ParquetFormatter,
    HuggingFaceFormatter,
    CSVFormatter,
)
from .analyzers import DatasetAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a single document in the dataset"""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0
    language: str = ""
    tokens: Optional[int] = None


@dataclass
class Dataset:
    """Represents a complete dataset"""
    documents: List[Document] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def __len__(self):
        return len(self.documents)

    def to_jsonl(self, path: str, **kwargs):
        """Export dataset to JSONL format"""
        formatter = JSONLFormatter()
        formatter.format(self, path, **kwargs)

    def to_parquet(self, path: str, **kwargs):
        """Export dataset to Parquet format"""
        formatter = ParquetFormatter()
        formatter.format(self, path, **kwargs)

    def to_huggingface(self, name: str, **kwargs):
        """Export dataset to HuggingFace format"""
        formatter = HuggingFaceFormatter()
        formatter.format(self, name, **kwargs)

    def to_csv(self, path: str, **kwargs):
        """Export dataset to CSV format"""
        formatter = CSVFormatter()
        formatter.format(self, path, **kwargs)


class Pipeline:
    """Processing pipeline for dataset creation"""

    def __init__(self):
        self.steps: List[Callable] = []
        self.config: Dict[str, Any] = {}

    def add_step(self, step: Callable, **kwargs):
        """Add a processing step to the pipeline"""
        self.steps.append((step, kwargs))
        return self

    def process(self, data: Any) -> Any:
        """Execute the pipeline on data"""
        result = data
        for step, kwargs in self.steps:
            if callable(step):
                result = step(result, **kwargs)
            else:
                result = step.process(result, **kwargs)
        return result

    def clear(self):
        """Clear all pipeline steps"""
        self.steps = []


class DatasetCreator:
    """Main class for creating AI training datasets"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the DatasetCreator

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_extractors()
        self._initialize_processors()
        self.sources = []
        self.pipeline = Pipeline()
        self.dataset = Dataset()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Load default config
            default_config_path = Path(__file__).parent.parent / "config" / "config.yaml"
            if default_config_path.exists():
                with open(default_config_path, 'r') as f:
                    return yaml.safe_load(f)
            return {}

    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('general', {}).get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _initialize_extractors(self):
        """Initialize data extractors"""
        self.extractors = {
            'pdf': PDFExtractor(self.config.get('extraction', {}).get('pdf', {})),
            'web': WebExtractor(self.config.get('extraction', {}).get('web', {})),
            'office': OfficeExtractor(self.config.get('extraction', {}).get('office', {})),
            'ebook': EbookExtractor(self.config.get('extraction', {}).get('ebook', {})),
            'text': TextExtractor({}),
        }

    def _initialize_processors(self):
        """Initialize text processors"""
        processing_config = self.config.get('processing', {})
        self.processors = {
            'chunker': Chunker(processing_config.get('chunking', {})),
            'cleaner': Cleaner(processing_config.get('cleaning', {})),
            'deduplicator': Deduplicator(processing_config.get('quality', {})),
            'privacy': PrivacyProtector(processing_config.get('privacy', {})),
            'quality': QualityFilter(processing_config.get('quality', {})),
        }

        # Add knowledge extractor if available
        if KNOWLEDGE_EXTRACTION_AVAILABLE:
            self.processors['knowledge'] = KnowledgeExtractor(
                processing_config.get('knowledge_extraction', {})
            )

        # Add metacognitive annotator if available
        if METACOGNITIVE_ANNOTATION_AVAILABLE:
            self.processors['metacognitive'] = MetacognitiveAnnotator(
                processing_config.get('metacognitive_annotation', {})
            )

        # Add adversarial tester if available
        if ADVERSARIAL_TESTING_AVAILABLE:
            self.processors['adversarial'] = AdversarialTester(
                processing_config.get('adversarial_testing', {})
            )

    def add_pdf(self, path: str, **kwargs):
        """Add a PDF file as a data source"""
        self.sources.append({
            'type': 'pdf',
            'path': path,
            'options': kwargs
        })
        logger.info(f"Added PDF source: {path}")
        return self

    def add_website(self, url: str, max_depth: int = 2, **kwargs):
        """Add a website as a data source"""
        self.sources.append({
            'type': 'web',
            'url': url,
            'max_depth': max_depth,
            'options': kwargs
        })
        logger.info(f"Added website source: {url} (max_depth={max_depth})")
        return self

    def add_directory(self, path: str, recursive: bool = True, pattern: str = "*", **kwargs):
        """Add all files in a directory as data sources"""
        path_obj = Path(path)
        if recursive:
            files = path_obj.rglob(pattern)
        else:
            files = path_obj.glob(pattern)

        for file_path in files:
            if file_path.is_file():
                file_type = self._detect_file_type(str(file_path))
                if file_type:
                    self.sources.append({
                        'type': file_type,
                        'path': str(file_path),
                        'options': kwargs
                    })

        logger.info(f"Added directory source: {path}")
        return self

    def add_office_document(self, path: str, **kwargs):
        """Add an Office document as a data source"""
        self.sources.append({
            'type': 'office',
            'path': path,
            'options': kwargs
        })
        logger.info(f"Added Office document source: {path}")
        return self

    def add_ebook(self, path: str, **kwargs):
        """Add an eBook as a data source"""
        self.sources.append({
            'type': 'ebook',
            'path': path,
            'options': kwargs
        })
        logger.info(f"Added eBook source: {path}")
        return self

    def _detect_file_type(self, path: str) -> Optional[str]:
        """Detect file type based on extension"""
        ext = Path(path).suffix.lower()
        type_map = {
            '.pdf': 'pdf',
            '.docx': 'office',
            '.doc': 'office',
            '.xlsx': 'office',
            '.xls': 'office',
            '.pptx': 'office',
            '.ppt': 'office',
            '.epub': 'ebook',
            '.mobi': 'ebook',
            '.azw3': 'ebook',
            '.txt': 'text',
            '.md': 'text',
            '.rtf': 'text',
        }
        return type_map.get(ext)

    def register_extractor(self, name: str, extractor):
        """Register a custom extractor"""
        self.extractors[name] = extractor
        logger.info(f"Registered custom extractor: {name}")

    def process(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        remove_pii: bool = True,
        quality_threshold: float = 0.7,
        remove_duplicates: bool = True,
        pipeline: Optional[Pipeline] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        chunking_strategy: str = 'semantic',
        extract_knowledge: bool = True,
        add_metacognitive_annotations: bool = True,
        enable_adversarial_testing: bool = True,
    ) -> Dataset:
        """Process all sources and create the dataset

        Args:
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            remove_pii: Whether to remove PII
            quality_threshold: Minimum quality score
            remove_duplicates: Whether to remove duplicates
            pipeline: Custom processing pipeline
            parallel: Whether to process in parallel
            max_workers: Maximum number of workers

        Returns:
            Processed Dataset object
        """
        logger.info(f"Starting dataset processing with {len(self.sources)} sources")

        # Setup pipeline if not provided
        if pipeline is None:
            pipeline = self._build_default_pipeline(
                chunk_size=chunk_size,
                overlap=overlap,
                remove_pii=remove_pii,
                quality_threshold=quality_threshold,
                remove_duplicates=remove_duplicates,
                chunking_strategy=chunking_strategy,
                extract_knowledge=extract_knowledge,
                add_metacognitive_annotations=add_metacognitive_annotations,
                enable_adversarial_testing=enable_adversarial_testing
            )

        # Extract data from all sources
        documents = []
        total_sources = len(self.sources)

        if parallel and total_sources > 1:
            max_workers = max_workers or self.config.get('general', {}).get('max_workers', 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for source in self.sources:
                    future = executor.submit(self._extract_source, source, progress_callback)
                    futures.append(future)

                for i, future in enumerate(tqdm(futures, desc="Extracting sources")):
                    if progress_callback:
                        # Update progress during extraction (30-60% range)
                        progress = 30 + int((i / max(total_sources, 1)) * 30)
                        source_name = self.sources[i].get('url', self.sources[i].get('path', 'source'))
                        progress_callback(progress, f"Extracting from {source_name[:50]}...")
                    doc = future.result()
                    if doc:
                        documents.append(doc)
        else:
            for i, source in enumerate(tqdm(self.sources, desc="Extracting sources")):
                if progress_callback:
                    # Update progress during extraction (30-60% range)
                    progress = 30 + int((i / max(total_sources, 1)) * 30)
                    source_name = source.get('url', source.get('path', 'source'))
                    progress_callback(progress, f"Extracting from {source_name[:50]}...")
                doc = self._extract_source(source, progress_callback)
                if doc:
                    documents.append(doc)

        # Process documents through pipeline
        logger.info("Processing documents through pipeline")
        processed_documents = []
        total_docs = len(documents)

        for i, doc in enumerate(documents):
            if progress_callback:
                # Calculate progress 60-80% range for processing
                progress = 60 + int((i / max(total_docs, 1)) * 20)
                doc_name = doc.source.split('/')[-1] if '/' in doc.source else doc.source
                progress_callback(progress, f"Processing document {i+1}/{total_docs}: {doc_name[:50]}...")

            processed_doc = pipeline.process(doc)
            if processed_doc:
                processed_documents.append(processed_doc)

        # Create dataset
        self.dataset = Dataset(
            documents=processed_documents,
            metadata={
                'created_at': datetime.now().isoformat(),
                'num_sources': len(self.sources),
                'processing_config': {
                    'chunk_size': chunk_size,
                    'overlap': overlap,
                    'remove_pii': remove_pii,
                    'quality_threshold': quality_threshold,
                    'remove_duplicates': remove_duplicates,
                }
            },
            config=self.config
        )

        # Calculate statistics
        analyzer = DatasetAnalyzer()
        self.dataset.statistics = analyzer.analyze(self.dataset)

        logger.info(f"Dataset created with {len(self.dataset)} documents")
        return self.dataset

    def _extract_source(self, source: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Optional[Document]:
        """Extract data from a single source"""
        try:
            source_type = source['type']
            extractor = self.extractors.get(source_type)

            if not extractor:
                logger.warning(f"No extractor found for type: {source_type}")
                return None

            # Extract data
            if source_type == 'web':
                # Pass max_depth and progress_callback for web extraction
                options = source.get('options', {})
                if 'max_depth' in source:
                    options['max_depth'] = source['max_depth']
                if progress_callback:
                    options['progress_callback'] = progress_callback
                data = extractor.extract(source['url'], **options)
            else:
                data = extractor.extract(source['path'], **source.get('options', {}))

            # Create document
            doc_id = hashlib.sha256(
                json.dumps(source, sort_keys=True).encode()
            ).hexdigest()[:16]

            document = Document(
                id=doc_id,
                text=data.get('text', ''),
                metadata=data.get('metadata', {}),
                source=source.get('path', source.get('url', '')),
            )

            return document

        except Exception as e:
            logger.error(f"Error extracting source {source}: {e}")
            return None

    def _build_default_pipeline(self, **kwargs) -> Pipeline:
        """Build default processing pipeline with advanced features"""
        pipeline = Pipeline()

        # Add cleaning step
        pipeline.add_step(self.processors['cleaner'])

        # Add quality filter
        if kwargs.get('quality_threshold'):
            pipeline.add_step(
                self.processors['quality'],
                threshold=kwargs['quality_threshold']
            )

        # Add privacy protection
        if kwargs.get('remove_pii'):
            pipeline.add_step(self.processors['privacy'])

        # Add knowledge extraction (before chunking to preserve context)
        if kwargs.get('extract_knowledge', True) and 'knowledge' in self.processors:
            pipeline.add_step(self.processors['knowledge'])

        # Add metacognitive annotations (after knowledge extraction)
        if kwargs.get('add_metacognitive_annotations', True) and 'metacognitive' in self.processors:
            pipeline.add_step(self.processors['metacognitive'])

        # Add chunking (now with semantic awareness)
        chunking_strategy = kwargs.get('chunking_strategy', 'semantic')
        pipeline.add_step(
            self.processors['chunker'],
            size=kwargs.get('chunk_size', 512),
            overlap=kwargs.get('overlap', 50),
            strategy=chunking_strategy
        )

        # Add deduplication
        if kwargs.get('remove_duplicates'):
            pipeline.add_step(self.processors['deduplicator'])

        # Add adversarial testing (final step to catch issues before output)
        if kwargs.get('enable_adversarial_testing', True) and 'adversarial' in self.processors:
            pipeline.add_step(self.processors['adversarial'])

        return pipeline

    def save_config(self, path: str):
        """Save current configuration to file"""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Configuration saved to {path}")

    def clear(self):
        """Clear all sources and reset dataset"""
        self.sources = []
        self.dataset = Dataset()
        self.pipeline = Pipeline()
        logger.info("Cleared all sources and reset dataset")