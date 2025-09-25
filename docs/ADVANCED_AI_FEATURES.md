# Advanced AI-Optimized Features

## Overview

Dataseter now includes state-of-the-art AI features designed to create superior training datasets for large language models and other AI systems. These features leverage transformer models, natural language processing, and advanced analysis techniques to ensure your datasets are optimized for AI training.

## Core Features

### 1. Semantic Chunking with Reasoning Preservation

Traditional chunking methods often break logical arguments and reasoning chains. Our semantic chunking:

- **Preserves Logical Flow**: Maintains causal relationships (because, therefore, thus)
- **Respects Argument Structure**: Keeps premises with their conclusions
- **Semantic Boundaries**: Uses sentence transformers to find natural breaks
- **Adaptive Sizing**: Adjusts chunk boundaries based on content coherence

```python
from dataseter import DatasetCreator

creator = DatasetCreator()
dataset = creator.process(
    chunking_strategy='semantic',  # Enable semantic chunking
    chunk_size=512,                # Target size
    preserve_reasoning=True         # Maintain logical chains
)
```

### 2. Knowledge Graph Extraction

Automatically builds structured knowledge from unstructured text:

- **Entity Recognition**: People, places, organizations, dates
- **Relationship Mapping**: Subject-predicate-object triples
- **Concept Hierarchies**: Domain taxonomies and ontologies
- **Fact Extraction**: Structured facts with confidence scores

```python
dataset = creator.process(
    extract_knowledge=True,
    knowledge_config={
        'extract_entities': True,
        'extract_relations': True,
        'extract_concepts': True,
        'confidence_threshold': 0.8
    }
)
```

### 3. Advanced Semantic Quality Scoring

Multi-dimensional quality assessment beyond simple heuristics:

- **Authority Score**: Identifies authoritative vs. speculative content
- **Coherence Analysis**: Measures logical flow and consistency
- **Complexity Assessment**: Evaluates cognitive load
- **Factuality Score**: Estimates factual accuracy
- **Training Value**: Predicts usefulness for AI training

```python
dataset = creator.process(
    quality_threshold=0.7,
    quality_dimensions={
        'authority_weight': 0.3,
        'coherence_weight': 0.25,
        'complexity_weight': 0.2,
        'factuality_weight': 0.15,
        'reasoning_weight': 0.1
    }
)
```

### 4. Metacognitive Annotations

Adds learning-focused metadata for educational AI applications:

- **Confidence Levels**: Statement certainty analysis
- **Prerequisite Knowledge**: Required background concepts
- **Learning Objectives**: Educational goals and outcomes
- **Cognitive Load**: Mental effort requirements
- **Difficulty Levels**: Content complexity classification

```python
dataset = creator.process(
    add_metacognitive_annotations=True,
    metacognitive_config={
        'confidence_analysis': True,
        'complexity_analysis': True,
        'prerequisite_analysis': True,
        'learning_objectives': True,
        'cognitive_load_assessment': True
    }
)
```

### 5. Adversarial Testing Framework

Ensures dataset quality and fairness:

- **Bias Detection**: Gender, racial, cultural, political biases
- **Contradiction Finding**: Logical inconsistencies
- **Harmful Content Screening**: Toxicity and inappropriate content
- **Fairness Analysis**: Representation balance
- **Robustness Testing**: Edge case identification

```python
dataset = creator.process(
    enable_adversarial_testing=True,
    adversarial_config={
        'bias_detection': True,
        'contradiction_detection': True,
        'harmful_content_detection': True,
        'fairness_analysis': True,
        'robustness_testing': True
    }
)
```

## Complete Example

```python
from dataseter import DatasetCreator

# Initialize with configuration
creator = DatasetCreator('config.yaml')

# Add diverse sources
creator.add_pdf('research_papers/*.pdf')
creator.add_website('https://docs.example.com', max_depth=3)
creator.add_directory('./textbooks', recursive=True)

# Process with all advanced features
dataset = creator.process(
    # Basic parameters
    chunk_size=512,
    overlap=50,
    remove_pii=True,
    quality_threshold=0.7,

    # Advanced AI features
    chunking_strategy='semantic',
    extract_knowledge=True,
    add_metacognitive_annotations=True,
    enable_adversarial_testing=True,

    # Parallel processing
    parallel=True,
    max_workers=4
)

# Analyze results
print(f"Created {len(dataset)} documents")
print(f"Knowledge graph entities: {dataset.statistics['entities_count']}")
print(f"Average quality score: {dataset.statistics['avg_quality_score']}")
print(f"Bias incidents detected: {dataset.statistics['bias_count']}")

# Export for AI training
dataset.to_huggingface('my-ai-dataset')
dataset.to_jsonl('output/training_data.jsonl')
```

## Output Format

Each processed document includes:

```json
{
  "text": "The processed text content...",
  "metadata": {
    "source": "document.pdf",
    "quality_score": 0.85,
    "language": "en"
  },
  "chunks": [
    {
      "text": "Chunk content...",
      "reasoning_chains": ["causal", "explanatory"],
      "coherence_score": 0.92,
      "entities": ["Einstein", "relativity theory"],
      "knowledge_graph": {
        "entities": [...],
        "relations": [...],
        "concepts": [...]
      },
      "quality_analysis": {
        "authority_score": 0.8,
        "coherence_score": 0.9,
        "complexity_score": 0.6,
        "factuality_score": 0.85,
        "training_value": 0.83
      },
      "metacognitive_annotations": {
        "confidence_level": "high",
        "complexity": "intermediate",
        "prerequisites": ["basic physics"],
        "cognitive_load": 0.65
      },
      "adversarial_analysis": {
        "bias_score": 0.1,
        "bias_types": [],
        "contradiction_score": 0.05,
        "toxicity_score": 0.02,
        "requires_review": false
      }
    }
  ]
}
```

## Performance Considerations

### Model Loading
- First run downloads required transformer models (~1-2GB)
- Models are cached for subsequent runs
- Use `TRANSFORMERS_CACHE` environment variable to set cache location

### Memory Requirements
- Minimum 8GB RAM recommended
- 16GB+ for large datasets
- GPU acceleration supported (CUDA, MPS)

### Processing Speed
- Semantic chunking: ~100-200 docs/minute
- Knowledge extraction: ~50-100 docs/minute
- Full pipeline: ~30-50 docs/minute

### Fallback Modes
All features have rule-based fallbacks when transformer models are unavailable:
- Semantic chunking → Rule-based sentence splitting
- Knowledge extraction → Keyword extraction
- Quality scoring → Heuristic scoring
- Adversarial testing → Pattern matching

## Configuration

### Via YAML Config File

```yaml
processing:
  chunking:
    strategy: semantic
    target_size: 512
    preserve_reasoning: true
    detect_arguments: true

  knowledge_extraction:
    extract_entities: true
    extract_relations: true
    extract_concepts: true
    min_confidence: 0.7

  semantic_quality:
    authority_weight: 0.3
    coherence_weight: 0.25
    complexity_weight: 0.2
    factuality_weight: 0.15
    reasoning_weight: 0.1

  metacognitive_annotation:
    confidence_analysis: true
    complexity_analysis: true
    prerequisite_analysis: true
    learning_objectives: true

  adversarial_testing:
    bias_detection: true
    contradiction_detection: true
    harmful_content_detection: true
    fairness_analysis: true
```

### Via Environment Variables

```bash
export DATASETER_CHUNKING_STRATEGY=semantic
export DATASETER_EXTRACT_KNOWLEDGE=true
export DATASETER_METACOGNITIVE=true
export DATASETER_ADVERSARIAL_TEST=true
export TRANSFORMERS_CACHE=/path/to/cache
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # For Mac M1/M2
```

## API Reference

### DatasetCreator.process()

```python
def process(
    self,
    # Standard parameters
    chunk_size: int = 512,
    overlap: int = 50,
    remove_pii: bool = True,
    quality_threshold: float = 0.7,
    remove_duplicates: bool = True,

    # Advanced AI parameters
    chunking_strategy: str = 'semantic',
    extract_knowledge: bool = True,
    add_metacognitive_annotations: bool = True,
    enable_adversarial_testing: bool = True,

    # Processing options
    pipeline: Optional[Pipeline] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable] = None
) -> Dataset:
    """
    Process sources with advanced AI features.

    Returns:
        Dataset object with processed documents and metadata
    """
```

## Testing

Run the comprehensive test suite:

```bash
# Test all advanced features
python tests/test_advanced_features.py

# Individual feature tests
python -m pytest tests/test_advanced_features.py::TestAdvancedFeatures::test_semantic_chunking
python -m pytest tests/test_advanced_features.py::TestAdvancedFeatures::test_knowledge_extraction
python -m pytest tests/test_advanced_features.py::TestAdvancedFeatures::test_adversarial_testing
```

## Troubleshooting

### Out of Memory Errors
- Reduce batch size: `export BATCH_SIZE=8`
- Disable GPU: `export CUDA_VISIBLE_DEVICES=-1`
- Use CPU-only models: `pip install dataseter[cpu]`

### Slow Processing
- Enable parallel processing: `parallel=True`
- Increase workers: `max_workers=8`
- Use faster models: `model_size='small'`

### Model Download Issues
- Check internet connection
- Set proxy if needed: `export HTTP_PROXY=...`
- Use offline mode with pre-downloaded models

## License & Attribution

These advanced features use open-source models:
- Sentence Transformers (Apache 2.0)
- Hugging Face Transformers (Apache 2.0)
- spaCy (MIT)

## Support

For issues or questions about advanced features:
- GitHub Issues: https://github.com/mkupermann/dataseter/issues
- Documentation: https://dataseter.readthedocs.io
- Examples: `/examples/advanced_features/`