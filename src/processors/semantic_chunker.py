"""
Advanced Semantic Chunking with Transformer-Based Boundary Detection
Implements AI-optimized chunking that preserves reasoning chains and conceptual completeness
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    from ..utils.spacy_loader import load_spacy_model, is_spacy_available
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Advanced semantic chunking for AI training datasets"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.chunk_size = config.get('target_size', 512)
        self.min_chunk_size = config.get('min_size', 100)
        self.max_chunk_size = config.get('max_size', 1024)
        self.coherence_threshold = config.get('coherence_threshold', 0.7)
        self.reasoning_preservation = config.get('preserve_reasoning', True)
        self.argument_detection = config.get('detect_arguments', True)

        # Initialize models if available
        self._init_models()

    def _init_models(self):
        """Initialize transformer models for semantic analysis"""
        self.sentence_model = None
        self.ner_pipeline = None
        self.nlp = None

        if TRANSFORMERS_AVAILABLE:
            try:
                # Lightweight but effective sentence embedding model
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

                # NER pipeline for entity detection
                self.ner_pipeline = pipeline("ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple")

                logger.info("Transformer models loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load transformer models: {e}")
                # Don't reassign TRANSFORMERS_AVAILABLE here - it's a module-level variable

        if SPACY_AVAILABLE:
            self.nlp = load_spacy_model("en_core_web_sm")
            if self.nlp:
                logger.info("SpaCy model loaded successfully")
            else:
                logger.warning("No SpaCy models available, falling back to rule-based methods")

    def process(self, document: Any, **kwargs) -> Any:
        """Process document with semantic chunking"""
        if hasattr(document, 'text'):
            chunks = self.semantic_chunk(document.text, **kwargs)
            document.chunks = chunks

            # Add semantic metadata
            if hasattr(document, 'metadata'):
                document.metadata['chunking_strategy'] = 'semantic'
                document.metadata['semantic_coherence'] = self._calculate_coherence_score(chunks)

        return document

    def semantic_chunk(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Perform semantic chunking with reasoning preservation"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, falling back to rule-based chunking")
            return self._rule_based_semantic_chunking(text, **kwargs)

        try:
            # Step 1: Split into sentences and analyze
            sentences = self._extract_sentences(text)
            if len(sentences) < 2:
                return [{'text': text, 'type': 'single_sentence', 'reasoning_chains': []}]

            # Step 2: Generate embeddings for semantic similarity
            try:
                embeddings = self.sentence_model.encode(sentences)
            except RuntimeError as e:
                if "MPS backend out of memory" in str(e):
                    logger.warning("MPS memory exhausted, falling back to rule-based chunking")
                    self.sentence_model = None
                    self.ner_model = None
                    return self._rule_based_semantic_chunking(text, **kwargs)
                raise

            # Step 3: Detect reasoning chains and argument structures
            reasoning_chains = self._detect_reasoning_chains(sentences, embeddings)
            argument_structures = self._detect_argument_structures(sentences)

            # Step 4: Find semantic boundaries
            boundaries = self._find_semantic_boundaries(sentences, embeddings)

            # Step 5: Preserve reasoning chains across boundaries
            adjusted_boundaries = self._preserve_reasoning_chains(boundaries, reasoning_chains)

            # Step 6: Create chunks with metadata
            chunks = self._create_semantic_chunks(sentences, adjusted_boundaries,
                                                reasoning_chains, argument_structures)

            return chunks

        except Exception as e:
            logger.error(f"Semantic chunking failed: {e}")
            return self._rule_based_semantic_chunking(text, **kwargs)

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences with proper handling of edge cases"""
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback to regex-based sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _detect_reasoning_chains(self, sentences: List[str], embeddings: np.ndarray) -> List[Dict]:
        """Detect reasoning chains and logical argument structures"""
        reasoning_chains = []

        # Patterns that indicate reasoning relationships
        reasoning_indicators = {
            'causal': [r'\bbecause\b', r'\bsince\b', r'\bas a result\b', r'\btherefore\b',
                      r'\bconsequently\b', r'\bthus\b', r'\bhence\b'],
            'contrast': [r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
                        r'\bon the other hand\b', r'\bconversely\b'],
            'addition': [r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
                        r'\bin addition\b', r'\balso\b'],
            'explanation': [r'\bthat is\b', r'\bin other words\b', r'\bfor example\b',
                           r'\bfor instance\b', r'\bspecifically\b'],
            'conclusion': [r'\bin conclusion\b', r'\bto summarize\b', r'\boverall\b',
                          r'\bin summary\b', r'\bfinally\b']
        }

        current_chain = []
        chain_type = None

        for i, sentence in enumerate(sentences):
            # Check for reasoning indicators
            found_indicator = None
            for indicator_type, patterns in reasoning_indicators.items():
                for pattern in patterns:
                    if re.search(pattern, sentence.lower()):
                        found_indicator = indicator_type
                        break
                if found_indicator:
                    break

            if found_indicator:
                if current_chain and chain_type == found_indicator:
                    # Continue current chain
                    current_chain.append(i)
                else:
                    # Start new chain
                    if current_chain and len(current_chain) > 1:
                        reasoning_chains.append({
                            'type': chain_type,
                            'sentence_indices': current_chain.copy(),
                            'coherence': self._calculate_chain_coherence(current_chain, embeddings)
                        })
                    current_chain = [i-1 if i > 0 else i, i]  # Include previous sentence
                    chain_type = found_indicator
            elif current_chain:
                # Check semantic similarity to continue chain
                if i > 0 and len(current_chain) > 0:
                    similarity = cosine_similarity([embeddings[current_chain[-1]]], [embeddings[i]])[0][0]
                    if similarity > self.coherence_threshold:
                        current_chain.append(i)
                    else:
                        # End current chain
                        if len(current_chain) > 1:
                            reasoning_chains.append({
                                'type': chain_type,
                                'sentence_indices': current_chain.copy(),
                                'coherence': self._calculate_chain_coherence(current_chain, embeddings)
                            })
                        current_chain = []
                        chain_type = None

        # Add final chain if exists
        if current_chain and len(current_chain) > 1:
            reasoning_chains.append({
                'type': chain_type,
                'sentence_indices': current_chain,
                'coherence': self._calculate_chain_coherence(current_chain, embeddings)
            })

        return reasoning_chains

    def _detect_argument_structures(self, sentences: List[str]) -> List[Dict]:
        """Detect argument structures (claim-evidence-conclusion)"""
        arguments = []

        # Simple patterns for argument detection
        claim_patterns = [r'\bi argue that\b', r'\bi claim\b', r'\bmy position is\b',
                         r'\bit is evident that\b', r'\bclearly\b']
        evidence_patterns = [r'\baccording to\b', r'\bstudies show\b', r'\bresearch indicates\b',
                            r'\bdata suggests\b', r'\bevidence shows\b']
        conclusion_patterns = [r'\btherefore\b', r'\bthus\b', r'\bin conclusion\b',
                              r'\bthis proves\b', r'\bhence\b']

        current_argument = {'claim': [], 'evidence': [], 'conclusion': []}

        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()

            # Check for claim indicators
            if any(re.search(pattern, sentence_lower) for pattern in claim_patterns):
                if current_argument['claim'] or current_argument['evidence']:
                    # Save previous argument if it has content
                    if current_argument['claim'] and current_argument['evidence']:
                        arguments.append(current_argument.copy())
                    current_argument = {'claim': [i], 'evidence': [], 'conclusion': []}
                else:
                    current_argument['claim'].append(i)

            # Check for evidence indicators
            elif any(re.search(pattern, sentence_lower) for pattern in evidence_patterns):
                current_argument['evidence'].append(i)

            # Check for conclusion indicators
            elif any(re.search(pattern, sentence_lower) for pattern in conclusion_patterns):
                current_argument['conclusion'].append(i)

        # Add final argument if complete
        if current_argument['claim'] and current_argument['evidence']:
            arguments.append(current_argument)

        return arguments

    def _find_semantic_boundaries(self, sentences: List[str], embeddings: np.ndarray) -> List[int]:
        """Find optimal boundaries based on semantic similarity"""
        if len(sentences) <= 2:
            return [len(sentences)]

        similarities = []
        for i in range(len(sentences) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)

        # Find local minima in similarity (potential boundaries)
        boundaries = []
        current_chunk_size = 0

        for i, similarity in enumerate(similarities):
            current_chunk_size += len(sentences[i].split())

            # Check if this is a good boundary point
            is_local_minimum = (
                i > 0 and i < len(similarities) - 1 and
                similarity < similarities[i-1] and
                similarity < similarities[i+1]
            ) or similarity < (np.mean(similarities) - np.std(similarities))

            # Consider chunk size constraints
            size_appropriate = (
                current_chunk_size >= self.min_chunk_size and
                current_chunk_size <= self.max_chunk_size
            )

            if is_local_minimum and size_appropriate:
                boundaries.append(i + 1)  # Boundary after sentence i
                current_chunk_size = 0

        # Ensure we don't exceed max chunk size
        if current_chunk_size > self.max_chunk_size:
            # Force boundary at appropriate location
            target_size = self.chunk_size
            current_size = 0
            for i, sentence in enumerate(sentences[boundaries[-1] if boundaries else 0:]):
                current_size += len(sentence.split())
                if current_size >= target_size:
                    boundaries.append((boundaries[-1] if boundaries else 0) + i + 1)
                    break

        # Always add final boundary
        if not boundaries or boundaries[-1] < len(sentences):
            boundaries.append(len(sentences))

        return sorted(set(boundaries))

    def _preserve_reasoning_chains(self, boundaries: List[int], reasoning_chains: List[Dict]) -> List[int]:
        """Adjust boundaries to preserve reasoning chains"""
        adjusted_boundaries = boundaries.copy()

        for chain in reasoning_chains:
            if chain['coherence'] > self.coherence_threshold:
                indices = chain['sentence_indices']
                start_idx, end_idx = min(indices), max(indices)

                # Check if any boundary splits this chain
                for boundary in boundaries:
                    if start_idx < boundary <= end_idx:
                        # Move boundary to preserve chain
                        if boundary - start_idx < end_idx - boundary:
                            # Move boundary before chain
                            new_boundary = start_idx
                        else:
                            # Move boundary after chain
                            new_boundary = end_idx + 1

                        if new_boundary not in adjusted_boundaries and new_boundary > 0:
                            adjusted_boundaries.remove(boundary)
                            adjusted_boundaries.append(new_boundary)

        return sorted(set(adjusted_boundaries))

    def _create_semantic_chunks(self, sentences: List[str], boundaries: List[int],
                               reasoning_chains: List[Dict], arguments: List[Dict]) -> List[Dict[str, Any]]:
        """Create final chunks with rich metadata"""
        chunks = []
        start_idx = 0

        for boundary in boundaries:
            chunk_sentences = sentences[start_idx:boundary]
            if not chunk_sentences:
                continue

            chunk_text = ' '.join(chunk_sentences)
            word_count = len(chunk_text.split())

            # Find reasoning chains in this chunk
            chunk_reasoning = []
            for chain in reasoning_chains:
                chain_indices = set(chain['sentence_indices'])
                chunk_indices = set(range(start_idx, boundary))
                if chain_indices.intersection(chunk_indices):
                    chunk_reasoning.append({
                        'type': chain['type'],
                        'coherence': chain['coherence'],
                        'sentence_count': len(chain_indices.intersection(chunk_indices))
                    })

            # Find arguments in this chunk
            chunk_arguments = []
            for arg in arguments:
                all_arg_indices = set(arg['claim'] + arg['evidence'] + arg['conclusion'])
                chunk_indices = set(range(start_idx, boundary))
                if all_arg_indices.intersection(chunk_indices):
                    chunk_arguments.append({
                        'has_claim': bool(set(arg['claim']).intersection(chunk_indices)),
                        'has_evidence': bool(set(arg['evidence']).intersection(chunk_indices)),
                        'has_conclusion': bool(set(arg['conclusion']).intersection(chunk_indices)),
                        'completeness': len(all_arg_indices.intersection(chunk_indices)) / len(all_arg_indices)
                    })

            # Extract entities if NER is available
            entities = []
            if self.ner_pipeline:
                try:
                    ner_results = self.ner_pipeline(chunk_text[:512])  # Limit for API
                    entities = [{'text': ent['word'], 'label': ent['entity_group'],
                               'confidence': ent['score']} for ent in ner_results]
                except Exception as e:
                    logger.debug(f"NER extraction failed: {e}")

            chunk = {
                'text': chunk_text,
                'sentence_count': len(chunk_sentences),
                'word_count': word_count,
                'start_sentence': start_idx,
                'end_sentence': boundary - 1,
                'reasoning_chains': chunk_reasoning,
                'arguments': chunk_arguments,
                'entities': entities,
                'semantic_type': self._classify_chunk_type(chunk_text),
                'complexity_score': self._calculate_complexity_score(chunk_text),
                'coherence_score': self._calculate_local_coherence(chunk_sentences)
            }

            chunks.append(chunk)
            start_idx = boundary

        return chunks

    def _calculate_coherence_score(self, chunks: List[Dict]) -> float:
        """Calculate overall coherence score for chunks"""
        if not chunks:
            return 0.0

        coherence_scores = [chunk.get('coherence_score', 0.5) for chunk in chunks]
        return np.mean(coherence_scores)

    def _calculate_chain_coherence(self, indices: List[int], embeddings: np.ndarray) -> float:
        """Calculate coherence of a reasoning chain"""
        if len(indices) < 2:
            return 1.0

        similarities = []
        for i in range(len(indices) - 1):
            sim = cosine_similarity([embeddings[indices[i]]], [embeddings[indices[i+1]]])[0][0]
            similarities.append(sim)

        return np.mean(similarities)

    def _classify_chunk_type(self, text: str) -> str:
        """Classify the type of content in the chunk"""
        text_lower = text.lower()

        # Check for different content types
        if any(word in text_lower for word in ['therefore', 'thus', 'because', 'since']):
            return 'reasoning'
        elif any(word in text_lower for word in ['for example', 'such as', 'instance']):
            return 'example'
        elif any(word in text_lower for word in ['define', 'definition', 'means', 'refers to']):
            return 'definition'
        elif any(word in text_lower for word in ['step', 'first', 'then', 'next', 'finally']):
            return 'procedural'
        elif any(word in text_lower for word in ['according to', 'study', 'research', 'data']):
            return 'evidence'
        else:
            return 'general'

    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate complexity score based on various factors"""
        words = text.split()
        sentences = len(re.split(r'[.!?]+', text))

        # Average words per sentence
        avg_words_per_sentence = len(words) / max(sentences, 1)

        # Lexical diversity
        unique_words = len(set(word.lower() for word in words))
        lexical_diversity = unique_words / len(words) if words else 0

        # Complex sentence indicators
        complex_indicators = len(re.findall(r'\b(however|although|because|therefore|consequently)\b', text.lower()))

        # Normalize to 0-1 scale
        complexity = (
            min(avg_words_per_sentence / 20, 1) * 0.4 +  # Sentence length
            lexical_diversity * 0.4 +                     # Vocabulary diversity
            min(complex_indicators / 10, 1) * 0.2         # Logical complexity
        )

        return complexity

    def _calculate_local_coherence(self, sentences: List[str]) -> float:
        """Calculate coherence within a chunk"""
        if len(sentences) < 2 or not self.sentence_model:
            return 0.5  # Default coherence

        try:
            embeddings = self.sentence_model.encode(sentences)
            similarities = []

            for i in range(len(sentences) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                similarities.append(sim)

            return np.mean(similarities)
        except:
            return 0.5

    def _rule_based_semantic_chunking(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Fallback rule-based semantic chunking when transformers unavailable"""
        # Enhanced version of the original semantic chunking
        sections = []

        # Split by multiple indicators
        section_patterns = [
            r'\n(?=[A-Z][^.]*:)',  # Headers with colons
            r'\n(?=[0-9]+\.)',      # Numbered lists
            r'\n(?=[-*])',          # Bullet points
            r'\n(?=[A-Z]{2,})',     # ALL CAPS headers
            r'\n\n+'                # Paragraph breaks
        ]

        current_sections = [text]
        for pattern in section_patterns:
            new_sections = []
            for section in current_sections:
                parts = re.split(pattern, section)
                new_sections.extend(parts)
            current_sections = [s.strip() for s in new_sections if s.strip()]

        chunks = []
        for i, section in enumerate(current_sections):
            words = section.split()
            if len(words) < self.min_chunk_size // 4:  # Adjust for word count
                continue

            chunk = {
                'text': section,
                'word_count': len(words),
                'type': 'rule_based_semantic',
                'reasoning_chains': [],
                'arguments': [],
                'entities': [],
                'semantic_type': self._classify_chunk_type(section),
                'complexity_score': self._calculate_complexity_score(section),
                'coherence_score': 0.5  # Default for rule-based
            }
            chunks.append(chunk)

        return chunks if chunks else [{'text': text, 'type': 'fallback', 'reasoning_chains': []}]