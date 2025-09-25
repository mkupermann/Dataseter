"""
Advanced Semantic Quality Scoring System
Evaluates content quality based on semantic coherence, authority, and AI training value
"""

import re
import math
import statistics
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import logging

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SEMANTIC_ANALYSIS_AVAILABLE = True
except ImportError:
    SEMANTIC_ANALYSIS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from ..utils.spacy_loader import load_spacy_model, is_spacy_available

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SemanticQualityScorer:
    """Advanced quality scoring based on multiple semantic factors"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.authority_weight = config.get('authority_weight', 0.3)
        self.coherence_weight = config.get('coherence_weight', 0.25)
        self.complexity_weight = config.get('complexity_weight', 0.2)
        self.factuality_weight = config.get('factuality_weight', 0.15)
        self.reasoning_weight = config.get('reasoning_weight', 0.1)

        self._init_models()
        self._init_authority_indicators()

    def _init_models(self):
        """Initialize models for quality assessment"""
        self.sentence_model = None
        self.nlp = None
        self.quality_classifier = None

        if SEMANTIC_ANALYSIS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer loaded for quality scoring")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")

        self.nlp = load_spacy_model("en_core_web_sm")
        if self.nlp:
            logger.info("SpaCy model loaded successfully")
        else:
            logger.warning("No SpaCy models available, falling back to rule-based methods")

        if TRANSFORMERS_AVAILABLE:
            try:
                # Quality classification pipeline (placeholder for actual quality model)
                self.quality_classifier = pipeline(
                    "text-classification",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
            except Exception as e:
                logger.debug(f"Quality classifier not available: {e}")

    def _init_authority_indicators(self):
        """Initialize indicators of content authority and credibility"""
        self.authority_indicators = {
            'academic': {
                'patterns': [
                    r'\b(?:research|study|experiment|analysis|investigation)\b',
                    r'\b(?:university|college|institute|laboratory)\b',
                    r'\b(?:professor|dr\.|ph\.d|researcher)\b',
                    r'\b(?:published|peer.reviewed|journal|conference)\b',
                    r'\b(?:doi|arxiv|pubmed)\b'
                ],
                'weight': 1.0
            },
            'expert': {
                'patterns': [
                    r'\b(?:expert|specialist|professional|authority)\b',
                    r'\b(?:ceo|director|chief|president|founder)\b',
                    r'\b(?:years? of experience|decades? of)\b',
                    r'\b(?:certified|licensed|accredited)\b'
                ],
                'weight': 0.8
            },
            'institutional': {
                'patterns': [
                    r'\b(?:government|official|ministry|department)\b',
                    r'\b(?:organization|association|foundation)\b',
                    r'\b(?:report|white paper|policy|guidelines)\b',
                    r'\b(?:statistics|census|survey|poll)\b'
                ],
                'weight': 0.7
            },
            'media': {
                'patterns': [
                    r'\b(?:news|journalist|reporter|correspondent)\b',
                    r'\b(?:article|story|coverage|investigation)\b',
                    r'\b(?:source|according to|reports)\b'
                ],
                'weight': 0.5
            },
            'opinion': {
                'patterns': [
                    r'\bi (?:think|believe|feel|opinion)\b',
                    r'\b(?:personally|my view|in my opinion)\b',
                    r'\b(?:seems|appears|might|could be)\b',
                    r'\b(?:blog|post|comment|review)\b'
                ],
                'weight': 0.2
            }
        }

        self.factuality_indicators = {
            'high_certainty': [
                r'\b(?:is|are|was|were|has|have|will)\b',
                r'\b(?:always|never|every|all|none)\b',
                r'\b\d+(?:\.\d+)?%\b',  # Percentages
                r'\b\d{4}\b',  # Years
                r'\b(?:fact|evidence|proof|demonstrate)\b'
            ],
            'medium_certainty': [
                r'\b(?:often|usually|typically|generally)\b',
                r'\b(?:most|many|some|few)\b',
                r'\b(?:tend to|likely|probable)\b'
            ],
            'low_certainty': [
                r'\b(?:might|could|may|perhaps|possibly)\b',
                r'\b(?:seem|appear|suggest|indicate)\b',
                r'\b(?:allegedly|reportedly|supposedly)\b'
            ]
        }

    def process(self, document: Any, **kwargs) -> Any:
        """Process document and add quality scores"""
        if hasattr(document, 'text'):
            # Score the full document
            quality_analysis = self.analyze_quality(document.text)
            document.quality_score = quality_analysis['overall_score']

            # Add detailed quality metadata
            if hasattr(document, 'metadata'):
                document.metadata['quality_analysis'] = quality_analysis
            else:
                document.quality_analysis = quality_analysis

            # Score individual chunks if they exist
            if hasattr(document, 'chunks'):
                for chunk in document.chunks:
                    if isinstance(chunk, dict) and 'text' in chunk:
                        chunk_quality = self.analyze_quality(chunk['text'])
                        chunk['quality_score'] = chunk_quality['overall_score']
                        chunk['quality_analysis'] = chunk_quality

        return document

    def analyze_quality(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive quality analysis"""
        analysis = {
            'overall_score': 0.0,
            'authority_score': 0.0,
            'coherence_score': 0.0,
            'complexity_score': 0.0,
            'factuality_score': 0.0,
            'reasoning_score': 0.0,
            'training_value': 0.0,
            'detailed_scores': {}
        }

        try:
            # Calculate component scores
            analysis['authority_score'] = self._score_authority(text)
            analysis['coherence_score'] = self._score_coherence(text)
            analysis['complexity_score'] = self._score_complexity(text)
            analysis['factuality_score'] = self._score_factuality(text)
            analysis['reasoning_score'] = self._score_reasoning(text)

            # Calculate weighted overall score
            analysis['overall_score'] = (
                analysis['authority_score'] * self.authority_weight +
                analysis['coherence_score'] * self.coherence_weight +
                analysis['complexity_score'] * self.complexity_weight +
                analysis['factuality_score'] * self.factuality_weight +
                analysis['reasoning_score'] * self.reasoning_weight
            )

            # Calculate training value (specific to AI training needs)
            analysis['training_value'] = self._calculate_training_value(analysis, text)

            # Add detailed breakdown
            analysis['detailed_scores'] = {
                'word_count': len(text.split()),
                'sentence_count': len(re.split(r'[.!?]+', text)),
                'avg_sentence_length': self._calculate_avg_sentence_length(text),
                'lexical_diversity': self._calculate_lexical_diversity(text),
                'readability': self._calculate_readability(text)
            }

        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            analysis['overall_score'] = 0.5  # Default score on failure

        return analysis

    def _score_authority(self, text: str) -> float:
        """Score content based on authority indicators"""
        text_lower = text.lower()
        total_score = 0.0
        total_weight = 0.0

        for category, data in self.authority_indicators.items():
            category_score = 0.0
            for pattern in data['patterns']:
                matches = len(re.findall(pattern, text_lower))
                category_score += matches

            # Normalize by text length and apply category weight
            text_length = len(text.split())
            normalized_score = min(category_score / max(text_length / 100, 1), 1.0)
            weighted_score = normalized_score * data['weight']

            total_score += weighted_score
            total_weight += data['weight']

        return min(total_score / total_weight if total_weight > 0 else 0.0, 1.0)

    def _score_coherence(self, text: str) -> float:
        """Score semantic coherence of the text"""
        if not SEMANTIC_ANALYSIS_AVAILABLE or not self.sentence_model:
            return self._score_coherence_fallback(text)

        try:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

            if len(sentences) < 2:
                return 0.5  # Default for short texts

            # Generate sentence embeddings
            embeddings = self.sentence_model.encode(sentences)

            # Calculate pairwise similarities
            similarities = []
            for i in range(len(sentences) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                similarities.append(sim)

            # Score based on consistency of similarities
            if similarities:
                mean_sim = np.mean(similarities)
                std_sim = np.std(similarities)

                # High coherence: high mean similarity, low standard deviation
                coherence_score = mean_sim - (std_sim * 0.5)
                return max(0.0, min(1.0, coherence_score))

        except Exception as e:
            logger.debug(f"Semantic coherence scoring failed: {e}")
            return self._score_coherence_fallback(text)

        return 0.5

    def _score_coherence_fallback(self, text: str) -> float:
        """Fallback coherence scoring without transformers"""
        # Use simpler heuristics
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.5

        # Check for transition words and coherence markers
        transition_words = [
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'nevertheless', 'meanwhile', 'similarly', 'likewise',
            'in contrast', 'on the other hand', 'as a result', 'for example'
        ]

        transition_count = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for transition in transition_words:
                if transition in sentence_lower:
                    transition_count += 1

        # Score based on transition density
        transition_density = transition_count / len(sentences)
        coherence_score = min(transition_density * 2, 0.8)  # Cap at 0.8 for fallback

        return coherence_score

    def _score_complexity(self, text: str) -> float:
        """Score text complexity appropriate for AI training"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not words or not sentences:
            return 0.0

        # Metrics for complexity
        avg_word_length = sum(len(word) for word in words) / len(words)
        avg_sentence_length = len(words) / len(sentences)

        # Lexical diversity (unique words / total words)
        lexical_diversity = len(set(word.lower() for word in words)) / len(words)

        # Syntactic complexity indicators
        complex_patterns = [
            r'\b(?:which|that|who|whom|whose)\b',  # Relative clauses
            r'\b(?:although|because|since|while|whereas)\b',  # Subordinate clauses
            r'[;:]',  # Complex punctuation
            r'\b(?:furthermore|moreover|however|nevertheless)\b'  # Complex connectors
        ]

        syntactic_complexity = sum(len(re.findall(pattern, text.lower()))
                                 for pattern in complex_patterns) / len(sentences)

        # Combine metrics (normalized to 0-1)
        complexity_score = (
            min(avg_word_length / 8, 1) * 0.3 +  # Average word length
            min(avg_sentence_length / 25, 1) * 0.3 +  # Average sentence length
            lexical_diversity * 0.2 +  # Vocabulary diversity
            min(syntactic_complexity / 2, 1) * 0.2  # Syntactic complexity
        )

        return max(0.0, min(1.0, complexity_score))

    def _score_factuality(self, text: str) -> float:
        """Score likelihood of factual content"""
        text_lower = text.lower()

        high_certainty_score = sum(len(re.findall(pattern, text_lower))
                                 for pattern in self.factuality_indicators['high_certainty'])
        medium_certainty_score = sum(len(re.findall(pattern, text_lower))
                                   for pattern in self.factuality_indicators['medium_certainty'])
        low_certainty_score = sum(len(re.findall(pattern, text_lower))
                                for pattern in self.factuality_indicators['low_certainty'])

        # Weight the scores
        weighted_score = (
            high_certainty_score * 1.0 +
            medium_certainty_score * 0.6 +
            low_certainty_score * (-0.3)  # Uncertainty reduces factuality score
        )

        # Normalize by text length
        text_length = len(text.split())
        normalized_score = weighted_score / max(text_length / 50, 1)

        # Check for specific factual elements
        factual_elements = [
            r'\b\d+(?:\.\d+)?%\b',  # Percentages
            r'\b\d{4}\b',  # Years
            r'\b(?:study|research|experiment|analysis)\b',  # Research indicators
            r'\b(?:according to|reports|statistics)\b'  # Attribution
        ]

        factual_density = sum(len(re.findall(pattern, text_lower))
                            for pattern in factual_elements) / max(text_length / 100, 1)

        # Combine scores
        factuality_score = min((normalized_score + factual_density) / 2, 1.0)
        return max(0.0, factuality_score)

    def _score_reasoning(self, text: str) -> float:
        """Score logical reasoning and argument structure"""
        text_lower = text.lower()

        # Reasoning indicators
        reasoning_patterns = {
            'causal': [r'\b(?:because|since|due to|as a result|therefore|thus|consequently)\b'],
            'evidence': [r'\b(?:evidence|proof|data|research|study)\b'],
            'comparison': [r'\b(?:however|but|although|whereas|in contrast|compared to)\b'],
            'conclusion': [r'\b(?:therefore|thus|in conclusion|overall|finally)\b'],
            'logical_structure': [r'\b(?:first|second|next|then|finally|lastly)\b']
        }

        reasoning_score = 0.0
        for category, patterns in reasoning_patterns.items():
            category_score = sum(len(re.findall(pattern, text_lower)) for pattern in patterns)
            reasoning_score += category_score * 0.2  # Each category worth 0.2

        # Normalize by text length
        text_length = len(text.split())
        normalized_reasoning = reasoning_score / max(text_length / 100, 1)

        # Check for argument structure (claim-evidence-conclusion)
        claim_indicators = len(re.findall(r'\b(?:argue|claim|assert|maintain)\b', text_lower))
        evidence_indicators = len(re.findall(r'\b(?:evidence|study|research|data)\b', text_lower))
        conclusion_indicators = len(re.findall(r'\b(?:therefore|thus|conclude|shows)\b', text_lower))

        # Bonus for complete argument structure
        if claim_indicators > 0 and evidence_indicators > 0 and conclusion_indicators > 0:
            normalized_reasoning += 0.3

        return max(0.0, min(1.0, normalized_reasoning))

    def _calculate_training_value(self, analysis: Dict, text: str) -> float:
        """Calculate specific value for AI training"""
        # Factors that make content valuable for AI training
        training_factors = {
            'explanation_quality': self._has_explanations(text),
            'example_richness': self._has_examples(text),
            'concept_density': self._calculate_concept_density(text),
            'reasoning_chains': analysis['reasoning_score'],
            'factual_accuracy': analysis['factuality_score'],
            'complexity_appropriateness': self._assess_complexity_for_training(analysis['complexity_score'])
        }

        # Weight the factors
        weights = {
            'explanation_quality': 0.25,
            'example_richness': 0.2,
            'concept_density': 0.2,
            'reasoning_chains': 0.15,
            'factual_accuracy': 0.1,
            'complexity_appropriateness': 0.1
        }

        training_value = sum(score * weights[factor]
                           for factor, score in training_factors.items())

        return max(0.0, min(1.0, training_value))

    def _has_explanations(self, text: str) -> float:
        """Check for explanatory content"""
        explanation_patterns = [
            r'\b(?:explain|because|reason|why|how|what|definition)\b',
            r'\b(?:means|refers to|defined as|in other words)\b',
            r'\b(?:for example|for instance|such as|like)\b',
            r'\b(?:that is|namely|specifically)\b'
        ]

        explanation_count = sum(len(re.findall(pattern, text.lower()))
                              for pattern in explanation_patterns)

        return min(explanation_count / max(len(text.split()) / 50, 1), 1.0)

    def _has_examples(self, text: str) -> float:
        """Check for examples and illustrations"""
        example_patterns = [
            r'\b(?:for example|for instance|such as|like|consider)\b',
            r'\b(?:e\.g\.|i\.e\.|viz\.)\b',
            r'\b(?:example|instance|illustration|case)\b'
        ]

        example_count = sum(len(re.findall(pattern, text.lower()))
                          for pattern in example_patterns)

        return min(example_count / max(len(text.split()) / 100, 1), 1.0)

    def _calculate_concept_density(self, text: str) -> float:
        """Calculate density of important concepts"""
        if not self.nlp:
            return 0.5  # Default when spaCy not available

        try:
            doc = self.nlp(text)

            # Extract noun phrases as concepts
            concepts = set()
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) > 1:  # Multi-word concepts
                    concepts.add(chunk.text.lower())

            # Extract named entities as concepts
            for ent in doc.ents:
                concepts.add(ent.text.lower())

            concept_density = len(concepts) / max(len(text.split()) / 20, 1)
            return min(concept_density, 1.0)

        except Exception as e:
            logger.debug(f"Concept density calculation failed: {e}")
            return 0.5

    def _assess_complexity_for_training(self, complexity_score: float) -> float:
        """Assess if complexity is appropriate for training"""
        # Optimal complexity for training is moderate (0.3-0.7)
        # Too simple or too complex reduces training value
        if 0.3 <= complexity_score <= 0.7:
            return 1.0  # Optimal range
        elif complexity_score < 0.3:
            return complexity_score / 0.3  # Scale up simple content
        else:  # complexity_score > 0.7
            return (1.0 - complexity_score) / 0.3  # Scale down complex content

    def _calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        total_words = sum(len(sentence.split()) for sentence in sentences)
        return total_words / len(sentences)

    def _calculate_lexical_diversity(self, text: str) -> float:
        """Calculate lexical diversity (TTR - Type Token Ratio)"""
        words = [word.lower() for word in text.split() if word.isalpha()]

        if not words:
            return 0.0

        unique_words = set(words)
        return len(unique_words) / len(words)

    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        syllables = sum(self._count_syllables(word) for word in words)

        if not sentences or not words:
            return 0.0

        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)

        # Simplified Flesch Reading Ease formula
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)

        # Normalize to 0-1 (typical Flesch scores range from 0-100)
        normalized_score = max(0, min(100, flesch_score)) / 100
        return normalized_score

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simple approximation)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_char_was_vowel = False

        for char in word:
            if char in vowels:
                if not previous_char_was_vowel:
                    syllable_count += 1
                previous_char_was_vowel = True
            else:
                previous_char_was_vowel = False

        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)  # Every word has at least one syllable