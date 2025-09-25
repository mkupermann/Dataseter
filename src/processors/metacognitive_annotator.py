"""
Metacognitive Annotations System
Adds confidence levels, complexity ratings, and prerequisite knowledge for AI training
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
import logging

try:
    import spacy
    from spacy.matcher import Matcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetacognitiveAnnotator:
    """Add metacognitive annotations to enhance AI training datasets"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.confidence_analysis = config.get('confidence_analysis', True)
        self.complexity_analysis = config.get('complexity_analysis', True)
        self.prerequisite_analysis = config.get('prerequisite_analysis', True)
        self.learning_objectives = config.get('learning_objectives', True)
        self.cognitive_load_assessment = config.get('cognitive_load_assessment', True)

        self._init_models()
        self._init_patterns()

    def _init_models(self):
        """Initialize models for metacognitive analysis"""
        self.nlp = None
        self.sentiment_analyzer = None
        self.matcher = None

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.matcher = Matcher(self.nlp.vocab)
                logger.info("SpaCy loaded for metacognitive analysis")
            except Exception as e:
                logger.warning(f"Failed to load SpaCy: {e}")

        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
            except Exception as e:
                logger.debug(f"Sentiment analyzer not available: {e}")

    def _init_patterns(self):
        """Initialize patterns for metacognitive analysis"""
        if not self.nlp:
            return

        # Confidence level patterns
        confidence_patterns = {
            'high_confidence': [
                [{"LEMMA": {"IN": ["be", "is", "are", "was", "were"]}}],
                [{"LEMMA": {"IN": ["know", "prove", "demonstrate", "establish"]}}],
                [{"LEMMA": "definite"}, {"LEMMA": {"IN": ["ly", ""]}}],
                [{"LEMMA": "certain"}, {"LEMMA": {"IN": ["ly", ""]}}],
                [{"LEMMA": {"IN": ["always", "never", "all", "every", "none"]}}]
            ],
            'medium_confidence': [
                [{"LEMMA": {"IN": ["likely", "probable", "tend", "usual"]}}],
                [{"LEMMA": {"IN": ["often", "frequently", "commonly"]}}],
                [{"LEMMA": {"IN": ["suggest", "indicate", "imply"]}}],
                [{"LEMMA": "general"}, {"LEMMA": {"IN": ["ly", ""]}}]
            ],
            'low_confidence': [
                [{"LEMMA": {"IN": ["might", "could", "may", "perhaps", "possibly"]}}],
                [{"LEMMA": {"IN": ["seem", "appear", "suspect", "guess"]}}],
                [{"LEMMA": {"IN": ["allegedly", "supposedly", "reportedly"]}}],
                [{"LEMMA": "uncertain"}, {"LEMMA": {"IN": ["ly", ""]}}]
            ]
        }

        # Add confidence patterns to matcher
        for confidence_level, patterns in confidence_patterns.items():
            for i, pattern in enumerate(patterns):
                self.matcher.add(f"{confidence_level}_{i}", [pattern])

        # Complexity indicators
        complexity_patterns = {
            'complex_syntax': [
                [{"DEP": "relcl"}],  # Relative clauses
                [{"DEP": "csubj"}],  # Clausal subjects
                [{"POS": "SCONJ"}],  # Subordinating conjunctions
            ],
            'technical_terms': [
                [{"POS": "NOUN", "ENT_TYPE": {"NOT_IN": ["PERSON", "ORG", "GPE"]}}],
                [{"LIKE_NUM": True}],
                [{"IS_ALPHA": True, "LENGTH": {">=": 8}}]  # Long words
            ]
        }

        for complexity_type, patterns in complexity_patterns.items():
            for i, pattern in enumerate(patterns):
                self.matcher.add(f"{complexity_type}_{i}", [pattern])

    def process(self, document: Any, **kwargs) -> Any:
        """Process document and add metacognitive annotations"""
        if hasattr(document, 'text'):
            annotations = self.analyze_metacognitive_aspects(document.text)

            # Add annotations to document
            if hasattr(document, 'metadata'):
                document.metadata['metacognitive_annotations'] = annotations
            else:
                document.metacognitive_annotations = annotations

            # Process chunks if they exist
            if hasattr(document, 'chunks'):
                for chunk in document.chunks:
                    if isinstance(chunk, dict) and 'text' in chunk:
                        chunk_annotations = self.analyze_metacognitive_aspects(chunk['text'])
                        chunk['metacognitive_annotations'] = chunk_annotations

        return document

    def analyze_metacognitive_aspects(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive metacognitive analysis"""
        annotations = {
            'confidence_distribution': {},
            'complexity_metrics': {},
            'prerequisite_knowledge': [],
            'learning_objectives': [],
            'cognitive_load': {},
            'pedagogical_features': {},
            'reasoning_requirements': []
        }

        try:
            if self.confidence_analysis:
                annotations['confidence_distribution'] = self._analyze_confidence(text)

            if self.complexity_analysis:
                annotations['complexity_metrics'] = self._analyze_complexity(text)

            if self.prerequisite_analysis:
                annotations['prerequisite_knowledge'] = self._identify_prerequisites(text)

            if self.learning_objectives:
                annotations['learning_objectives'] = self._identify_learning_objectives(text)

            if self.cognitive_load_assessment:
                annotations['cognitive_load'] = self._assess_cognitive_load(text)

            # Additional pedagogical features
            annotations['pedagogical_features'] = self._identify_pedagogical_features(text)
            annotations['reasoning_requirements'] = self._identify_reasoning_requirements(text)

        except Exception as e:
            logger.error(f"Metacognitive analysis failed: {e}")

        return annotations

    def _analyze_confidence(self, text: str) -> Dict[str, Any]:
        """Analyze confidence levels in the text"""
        confidence_analysis = {
            'overall_confidence': 0.5,
            'confidence_markers': {
                'high': 0,
                'medium': 0,
                'low': 0,
                'uncertain': 0
            },
            'certainty_phrases': [],
            'uncertainty_phrases': [],
            'confidence_score': 0.5
        }

        if not self.nlp:
            return self._analyze_confidence_fallback(text)

        try:
            doc = self.nlp(text)
            matches = self.matcher(doc)

            certainty_count = 0
            uncertainty_count = 0

            for match_id, start, end in matches:
                match_label = self.nlp.vocab.strings[match_id]
                span_text = doc[start:end].text

                if 'high_confidence' in match_label:
                    confidence_analysis['confidence_markers']['high'] += 1
                    confidence_analysis['certainty_phrases'].append(span_text)
                    certainty_count += 2
                elif 'medium_confidence' in match_label:
                    confidence_analysis['confidence_markers']['medium'] += 1
                    certainty_count += 1
                elif 'low_confidence' in match_label:
                    confidence_analysis['confidence_markers']['low'] += 1
                    confidence_analysis['uncertainty_phrases'].append(span_text)
                    uncertainty_count += 1

            # Calculate overall confidence
            total_markers = sum(confidence_analysis['confidence_markers'].values())
            if total_markers > 0:
                weighted_score = (
                    confidence_analysis['confidence_markers']['high'] * 1.0 +
                    confidence_analysis['confidence_markers']['medium'] * 0.6 +
                    confidence_analysis['confidence_markers']['low'] * 0.2
                ) / total_markers
                confidence_analysis['confidence_score'] = weighted_score
                confidence_analysis['overall_confidence'] = weighted_score
            else:
                # No explicit confidence markers - analyze implicit confidence
                confidence_analysis['confidence_score'] = self._implicit_confidence_analysis(text)
                confidence_analysis['overall_confidence'] = confidence_analysis['confidence_score']

        except Exception as e:
            logger.debug(f"Advanced confidence analysis failed: {e}")
            return self._analyze_confidence_fallback(text)

        return confidence_analysis

    def _analyze_confidence_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback confidence analysis using regex patterns"""
        confidence_analysis = {
            'overall_confidence': 0.5,
            'confidence_markers': {'high': 0, 'medium': 0, 'low': 0, 'uncertain': 0},
            'certainty_phrases': [],
            'uncertainty_phrases': [],
            'confidence_score': 0.5
        }

        high_confidence_patterns = [
            r'\b(?:is|are|will|must|always|never|definitely|certainly)\b',
            r'\b(?:proven|established|demonstrated|confirmed)\b',
            r'\b(?:fact|evidence|proof|undoubtedly)\b'
        ]

        low_confidence_patterns = [
            r'\b(?:might|could|may|perhaps|possibly|allegedly)\b',
            r'\b(?:seems?|appears?|suggests?|indicates?)\b',
            r'\b(?:probably|likely|presumably|supposedly)\b'
        ]

        text_lower = text.lower()

        for pattern in high_confidence_patterns:
            matches = re.findall(pattern, text_lower)
            confidence_analysis['confidence_markers']['high'] += len(matches)

        for pattern in low_confidence_patterns:
            matches = re.findall(pattern, text_lower)
            confidence_analysis['confidence_markers']['low'] += len(matches)

        # Calculate confidence score
        high_count = confidence_analysis['confidence_markers']['high']
        low_count = confidence_analysis['confidence_markers']['low']

        if high_count + low_count > 0:
            confidence_score = high_count / (high_count + low_count)
            confidence_analysis['confidence_score'] = confidence_score
            confidence_analysis['overall_confidence'] = confidence_score

        return confidence_analysis

    def _analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze multiple dimensions of text complexity"""
        complexity_metrics = {
            'lexical_complexity': 0.0,
            'syntactic_complexity': 0.0,
            'semantic_complexity': 0.0,
            'conceptual_complexity': 0.0,
            'overall_complexity': 0.0,
            'complexity_factors': {}
        }

        # Lexical complexity
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            unique_words = len(set(word.lower() for word in words))
            lexical_diversity = unique_words / len(words)

            # Complex word ratio (words > 6 characters)
            complex_words = sum(1 for word in words if len(word) > 6)
            complex_word_ratio = complex_words / len(words)

            complexity_metrics['lexical_complexity'] = min(
                (avg_word_length / 8) * 0.4 +
                lexical_diversity * 0.3 +
                complex_word_ratio * 0.3, 1.0
            )

        # Syntactic complexity
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            avg_sentence_length = len(words) / len(sentences) if words else 0

            # Count complex syntactic structures
            complex_structures = (
                len(re.findall(r'\b(?:which|that|who|whom|whose)\b', text.lower())) +  # Relative clauses
                len(re.findall(r'\b(?:although|because|since|while|whereas|if)\b', text.lower())) +  # Subordinate clauses
                len(re.findall(r'[;:]', text)) +  # Complex punctuation
                len(re.findall(r'\b(?:however|moreover|furthermore|nevertheless|consequently)\b', text.lower()))  # Complex connectors
            )

            syntactic_density = complex_structures / len(sentences)

            complexity_metrics['syntactic_complexity'] = min(
                (avg_sentence_length / 25) * 0.6 +
                (syntactic_density / 3) * 0.4, 1.0
            )

        # Semantic complexity (concept density and abstraction)
        if self.nlp:
            try:
                doc = self.nlp(text)

                # Abstract concepts (entities, complex noun phrases)
                abstract_concepts = 0
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) > 2:  # Multi-word concepts
                        abstract_concepts += 1

                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'PRODUCT', 'EVENT', 'LAW', 'LANGUAGE']:
                        abstract_concepts += 1

                semantic_density = abstract_concepts / len(sentences) if sentences else 0
                complexity_metrics['semantic_complexity'] = min(semantic_density / 2, 1.0)

            except Exception as e:
                logger.debug(f"Semantic complexity analysis failed: {e}")
                complexity_metrics['semantic_complexity'] = 0.5

        # Conceptual complexity (domain-specific terms, technical vocabulary)
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+(?:ology|ometry|ics|tion|sion|ism|ity)\b',  # Technical suffixes
            r'\b(?:analysis|synthesis|methodology|paradigm|framework)\b'  # Academic terms
        ]

        technical_density = sum(len(re.findall(pattern, text)) for pattern in technical_patterns)
        conceptual_complexity = min(technical_density / max(len(words) / 50, 1), 1.0) if words else 0
        complexity_metrics['conceptual_complexity'] = conceptual_complexity

        # Overall complexity
        complexity_metrics['overall_complexity'] = (
            complexity_metrics['lexical_complexity'] * 0.25 +
            complexity_metrics['syntactic_complexity'] * 0.25 +
            complexity_metrics['semantic_complexity'] * 0.25 +
            complexity_metrics['conceptual_complexity'] * 0.25
        )

        # Store detailed factors
        complexity_metrics['complexity_factors'] = {
            'avg_word_length': avg_word_length if words else 0,
            'avg_sentence_length': avg_sentence_length if sentences else 0,
            'lexical_diversity': lexical_diversity if words else 0,
            'technical_term_density': technical_density,
            'complex_structure_density': complex_structures / len(sentences) if sentences else 0
        }

        return complexity_metrics

    def _identify_prerequisites(self, text: str) -> List[Dict[str, Any]]:
        """Identify prerequisite knowledge required to understand the text"""
        prerequisites = []

        # Domain knowledge indicators
        domain_patterns = {
            'mathematics': [
                r'\b(?:equation|theorem|proof|derivative|integral|matrix|vector|algebra|calculus|geometry)\b',
                r'\b(?:mathematical|numeric|quantitative|statistical)\b'
            ],
            'science': [
                r'\b(?:hypothesis|experiment|theory|molecule|atom|cell|DNA|protein|chemical|physics|biology)\b',
                r'\b(?:scientific|empirical|research|laboratory)\b'
            ],
            'computer_science': [
                r'\b(?:algorithm|programming|software|database|network|artificial|intelligence|machine|learning)\b',
                r'\b(?:computational|digital|technical|system)\b'
            ],
            'business': [
                r'\b(?:strategy|management|marketing|finance|economics|profit|revenue|market|customer)\b',
                r'\b(?:business|commercial|corporate|organizational)\b'
            ],
            'history': [
                r'\b(?:historical|ancient|medieval|renaissance|revolution|war|empire|civilization)\b',
                r'\b\d{3,4}(?:\s?(?:BC|AD|CE|BCE))?\b'  # Years
            ],
            'philosophy': [
                r'\b(?:ethical|moral|philosophical|metaphysical|epistemological|ontological|logic|reasoning)\b',
                r'\b(?:argue|argument|premise|conclusion|validity|truth)\b'
            ]
        }

        text_lower = text.lower()

        for domain, patterns in domain_patterns.items():
            domain_score = 0
            matched_terms = []

            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                domain_score += len(matches)
                matched_terms.extend(matches)

            if domain_score > 0:
                # Calculate prerequisite strength based on term density
                prerequisite_strength = min(domain_score / max(len(text.split()) / 50, 1), 1.0)

                if prerequisite_strength > 0.1:  # Threshold for inclusion
                    prerequisites.append({
                        'domain': domain,
                        'strength': prerequisite_strength,
                        'matched_terms': list(set(matched_terms)),
                        'description': self._get_domain_description(domain)
                    })

        # Technical prerequisites based on complexity
        if any(p['domain'] in ['mathematics', 'science', 'computer_science'] for p in prerequisites):
            prerequisites.append({
                'domain': 'technical_literacy',
                'strength': 0.7,
                'matched_terms': [],
                'description': 'Basic technical and scientific literacy required'
            })

        # Reading level prerequisites
        complexity_score = self._calculate_reading_complexity(text)
        if complexity_score > 0.7:
            prerequisites.append({
                'domain': 'advanced_reading',
                'strength': complexity_score,
                'matched_terms': [],
                'description': 'Advanced reading comprehension skills required'
            })

        return sorted(prerequisites, key=lambda x: x['strength'], reverse=True)

    def _identify_learning_objectives(self, text: str) -> List[Dict[str, Any]]:
        """Identify potential learning objectives from the content"""
        objectives = []

        # Learning verb patterns
        learning_verbs = {
            'remember': [r'\b(?:define|identify|list|name|recall|recognize|state)\b'],
            'understand': [r'\b(?:explain|describe|interpret|summarize|classify|compare|discuss)\b'],
            'apply': [r'\b(?:apply|demonstrate|use|solve|implement|execute|operate)\b'],
            'analyze': [r'\b(?:analyze|examine|investigate|break down|differentiate|organize)\b'],
            'evaluate': [r'\b(?:evaluate|assess|judge|critique|justify|argue|defend)\b'],
            'create': [r'\b(?:create|design|develop|generate|produce|construct|formulate)\b']
        }

        text_lower = text.lower()

        for cognitive_level, patterns in learning_verbs.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    objectives.append({
                        'cognitive_level': cognitive_level,
                        'verbs_present': matches,
                        'bloom_taxonomy_level': self._get_bloom_level(cognitive_level),
                        'objective_strength': len(matches) / max(len(text.split()) / 100, 1)
                    })

        # Content-based objectives
        if self.nlp:
            try:
                doc = self.nlp(text)

                # Extract key concepts that could be learning objectives
                key_concepts = []
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) > 1 and chunk.root.pos_ == 'NOUN':
                        key_concepts.append(chunk.text)

                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'PRODUCT', 'EVENT', 'LAW']:
                        key_concepts.append(ent.text)

                if key_concepts:
                    objectives.append({
                        'cognitive_level': 'conceptual_knowledge',
                        'key_concepts': key_concepts[:5],  # Top 5 concepts
                        'bloom_taxonomy_level': 2,
                        'objective_strength': len(key_concepts) / 10
                    })

            except Exception as e:
                logger.debug(f"Concept extraction for objectives failed: {e}")

        return objectives

    def _assess_cognitive_load(self, text: str) -> Dict[str, Any]:
        """Assess cognitive load requirements for processing the text"""
        cognitive_load = {
            'intrinsic_load': 0.0,
            'extraneous_load': 0.0,
            'germane_load': 0.0,
            'total_load': 0.0,
            'load_factors': {}
        }

        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not words or not sentences:
            return cognitive_load

        # Intrinsic load (inherent difficulty of the material)
        # Based on concept density and abstraction
        technical_terms = len(re.findall(r'\b[A-Z]{2,}\b|\b\w+(?:ology|ometry|ics|tion)\b', text))
        concept_density = technical_terms / len(words)

        avg_word_length = sum(len(word) for word in words) / len(words)
        intrinsic_complexity = (concept_density * 0.6) + (min(avg_word_length / 8, 1) * 0.4)
        cognitive_load['intrinsic_load'] = min(intrinsic_complexity, 1.0)

        # Extraneous load (cognitive load from presentation/structure)
        # Based on sentence complexity and coherence
        avg_sentence_length = len(words) / len(sentences)
        complex_sentences = sum(1 for s in sentences if len(s.split()) > 20)
        sentence_complexity = complex_sentences / len(sentences)

        # Poor coherence increases extraneous load
        coherence_markers = len(re.findall(
            r'\b(?:however|therefore|furthermore|moreover|consequently|nevertheless)\b',
            text.lower()
        ))
        coherence_score = min(coherence_markers / len(sentences), 1.0)

        extraneous_complexity = (
            min(avg_sentence_length / 30, 1) * 0.4 +
            sentence_complexity * 0.3 +
            (1 - coherence_score) * 0.3  # Lower coherence = higher extraneous load
        )
        cognitive_load['extraneous_load'] = min(extraneous_complexity, 1.0)

        # Germane load (cognitive effort toward learning)
        # Based on explanatory content and examples
        explanatory_patterns = [
            r'\b(?:explain|because|reason|why|how|example|instance)\b',
            r'\b(?:means|refers to|defined as|that is)\b'
        ]

        explanatory_density = sum(
            len(re.findall(pattern, text.lower())) for pattern in explanatory_patterns
        ) / len(sentences)

        germane_complexity = min(explanatory_density / 2, 1.0)
        cognitive_load['germane_load'] = germane_complexity

        # Total cognitive load
        cognitive_load['total_load'] = (
            cognitive_load['intrinsic_load'] * 0.5 +
            cognitive_load['extraneous_load'] * 0.3 +
            cognitive_load['germane_load'] * 0.2
        )

        # Store load factors for analysis
        cognitive_load['load_factors'] = {
            'concept_density': concept_density,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'sentence_complexity_ratio': sentence_complexity,
            'coherence_score': coherence_score,
            'explanatory_density': explanatory_density
        }

        return cognitive_load

    def _identify_pedagogical_features(self, text: str) -> Dict[str, Any]:
        """Identify pedagogical features that support learning"""
        features = {
            'has_examples': False,
            'has_definitions': False,
            'has_analogies': False,
            'has_questions': False,
            'has_summaries': False,
            'scaffolding_elements': [],
            'engagement_features': []
        }

        text_lower = text.lower()

        # Examples
        example_patterns = [
            r'\b(?:for example|for instance|such as|like|consider)\b',
            r'\b(?:e\.g\.|i\.e\.)\b'
        ]
        features['has_examples'] = any(re.search(pattern, text_lower) for pattern in example_patterns)

        # Definitions
        definition_patterns = [
            r'\b(?:define|definition|means|refers to|defined as|is called)\b',
            r'\b(?:that is|namely|in other words)\b'
        ]
        features['has_definitions'] = any(re.search(pattern, text_lower) for pattern in definition_patterns)

        # Analogies
        analogy_patterns = [
            r'\b(?:like|similar to|analogous to|just as|comparable to)\b',
            r'\b(?:metaphor|analogy|comparison)\b'
        ]
        features['has_analogies'] = any(re.search(pattern, text_lower) for pattern in analogy_patterns)

        # Questions
        features['has_questions'] = '?' in text

        # Summaries
        summary_patterns = [
            r'\b(?:in summary|to summarize|in conclusion|overall)\b',
            r'\b(?:key points|main ideas|important)\b'
        ]
        features['has_summaries'] = any(re.search(pattern, text_lower) for pattern in summary_patterns)

        # Scaffolding elements
        scaffolding_patterns = {
            'step_by_step': r'\b(?:step|first|second|then|next|finally)\b',
            'prerequisites': r'\b(?:before|prerequisite|required|need to know)\b',
            'progressive_difficulty': r'\b(?:basic|advanced|complex|simple|difficult)\b'
        }

        for element, pattern in scaffolding_patterns.items():
            if re.search(pattern, text_lower):
                features['scaffolding_elements'].append(element)

        # Engagement features
        engagement_patterns = {
            'direct_address': r'\b(?:you|your|we|us|our)\b',
            'imperative_verbs': r'\b(?:consider|imagine|think|remember)\b',
            'rhetorical_questions': r'[?]',
            'exclamations': r'[!]'
        }

        for feature, pattern in engagement_patterns.items():
            if re.search(pattern, text_lower):
                features['engagement_features'].append(feature)

        return features

    def _identify_reasoning_requirements(self, text: str) -> List[str]:
        """Identify types of reasoning required to understand the text"""
        reasoning_types = []

        reasoning_patterns = {
            'deductive': [r'\b(?:therefore|thus|consequently|follows that|proves)\b'],
            'inductive': [r'\b(?:suggests|indicates|evidence shows|pattern|trend)\b'],
            'analogical': [r'\b(?:similar|like|analogous|comparable|parallel)\b'],
            'causal': [r'\b(?:because|caused by|results in|leads to|due to)\b'],
            'conditional': [r'\b(?:if|when|unless|provided that|assuming)\b'],
            'comparative': [r'\b(?:compared to|versus|different from|contrast)\b'],
            'evaluative': [r'\b(?:evaluate|assess|judge|critique|better|worse)\b']
        }

        text_lower = text.lower()

        for reasoning_type, patterns in reasoning_patterns.items():
            if any(re.search(pattern, text_lower) for pattern in patterns):
                reasoning_types.append(reasoning_type)

        return reasoning_types

    def _implicit_confidence_analysis(self, text: str) -> float:
        """Analyze implicit confidence markers"""
        # Check for hedging language
        hedging_patterns = [
            r'\b(?:somewhat|rather|quite|fairly|relatively)\b',
            r'\b(?:tend to|inclined to|appears to)\b'
        ]

        hedging_count = sum(len(re.findall(pattern, text.lower())) for pattern in hedging_patterns)

        # Check for assertive language
        assertive_patterns = [
            r'\b(?:clearly|obviously|undoubtedly|certainly)\b',
            r'\b(?:must|will|always|never)\b'
        ]

        assertive_count = sum(len(re.findall(pattern, text.lower())) for pattern in assertive_patterns)

        # Balance assertive vs hedging language
        total_markers = hedging_count + assertive_count
        if total_markers > 0:
            return assertive_count / total_markers
        else:
            return 0.5  # Neutral confidence when no markers

    def _get_domain_description(self, domain: str) -> str:
        """Get description for domain prerequisites"""
        descriptions = {
            'mathematics': 'Mathematical concepts, formulas, and quantitative reasoning',
            'science': 'Scientific principles, experimental methods, and research concepts',
            'computer_science': 'Programming concepts, algorithms, and technical systems',
            'business': 'Business terminology, strategic concepts, and market dynamics',
            'history': 'Historical context, chronological understanding, and cultural knowledge',
            'philosophy': 'Philosophical reasoning, logical arguments, and abstract concepts',
            'technical_literacy': 'General technical and scientific literacy',
            'advanced_reading': 'Advanced reading comprehension and vocabulary'
        }
        return descriptions.get(domain, f'{domain} knowledge and concepts')

    def _get_bloom_level(self, cognitive_level: str) -> int:
        """Get Bloom's taxonomy numeric level"""
        bloom_levels = {
            'remember': 1,
            'understand': 2,
            'apply': 3,
            'analyze': 4,
            'evaluate': 5,
            'create': 6
        }
        return bloom_levels.get(cognitive_level, 2)

    def _calculate_reading_complexity(self, text: str) -> float:
        """Calculate reading complexity score"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not words or not sentences:
            return 0.0

        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Simple complexity score based on sentence and word length
        complexity = (
            min(avg_sentence_length / 25, 1) * 0.6 +
            min(avg_word_length / 8, 1) * 0.4
        )

        return complexity