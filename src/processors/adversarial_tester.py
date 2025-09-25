"""
Adversarial Testing Framework for Bias Detection and Content Quality Assurance
Detects bias, contradictions, harmful content, and ensures dataset robustness for AI training
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
import logging

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SIMILARITY_ANALYSIS_AVAILABLE = True
except ImportError:
    SIMILARITY_ANALYSIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AdversarialTester:
    """Adversarial testing framework for dataset quality and bias detection"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.bias_detection = config.get('bias_detection', True)
        self.contradiction_detection = config.get('contradiction_detection', True)
        self.harmful_content_detection = config.get('harmful_content_detection', True)
        self.fairness_analysis = config.get('fairness_analysis', True)
        self.robustness_testing = config.get('robustness_testing', True)

        # Bias detection thresholds
        self.bias_threshold = config.get('bias_threshold', 0.7)
        self.contradiction_threshold = config.get('contradiction_threshold', 0.8)
        self.harm_threshold = config.get('harm_threshold', 0.6)

        self._init_models()
        self._init_bias_patterns()
        self._init_harmful_content_patterns()

    def _init_models(self):
        """Initialize models for adversarial testing"""
        self.nlp = None
        self.toxicity_classifier = None
        self.bias_classifier = None
        self.sentence_model = None

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("SpaCy loaded for adversarial testing")
            except Exception as e:
                logger.warning(f"Failed to load SpaCy: {e}")

        if TRANSFORMERS_AVAILABLE:
            try:
                # Toxicity/harmful content detection
                self.toxicity_classifier = pipeline(
                    "text-classification",
                    model="martin-ha/toxic-comment-model"
                )
                logger.info("Toxicity classifier loaded")
            except Exception as e:
                logger.debug(f"Toxicity classifier not available: {e}")

            try:
                # Bias detection classifier (if available)
                self.bias_classifier = pipeline(
                    "text-classification",
                    model="d4data/bias-detection-model"
                )
                logger.info("Bias classifier loaded")
            except Exception as e:
                logger.debug(f"Bias classifier not available: {e}")

        if SIMILARITY_ANALYSIS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence model loaded for contradiction detection")
            except Exception as e:
                logger.debug(f"Sentence model not available: {e}")

    def _init_bias_patterns(self):
        """Initialize bias detection patterns"""
        self.bias_patterns = {
            'gender_bias': {
                'patterns': [
                    # Stereotypical associations
                    r'\b(?:women|girls?|female)\s+(?:are\s+)?(?:naturally|typically|usually|often)\s+(?:more|less|better|worse)\s+(?:at|with|in)',
                    r'\b(?:men|boys?|male)\s+(?:are\s+)?(?:naturally|typically|usually|often)\s+(?:more|less|better|worse)\s+(?:at|with|in)',
                    # Role assumptions
                    r'\b(?:housewife|homemaker|stay-at-home\s+mom)\b',
                    r'\b(?:working\s+mother|career\s+woman)\b',
                    # Appearance-focused language
                    r'\b(?:attractive|beautiful|pretty|sexy)\s+(?:woman|girl|female)',
                    r'\b(?:strong|handsome|masculine)\s+(?:man|boy|male)'
                ],
                'weight': 0.8
            },
            'racial_bias': {
                'patterns': [
                    # Stereotypical descriptions
                    r'\b(?:asian|chinese|japanese)\s+(?:are\s+)?(?:good|bad|better|worse)\s+(?:at|with|in)\s+(?:math|science|driving)',
                    r'\b(?:black|african)\s+(?:people|person|man|woman)\s+(?:are\s+)?(?:more|less|naturally|typically)',
                    # Problematic generalizations
                    r'\b(?:white|caucasian)\s+(?:privilege|supremacy)\b',
                    r'\b(?:minority|ethnic)\s+(?:group|community)\s+(?:are|is|tend)',
                    # Cultural assumptions
                    r'\b(?:ghetto|hood|urban)\s+(?:culture|lifestyle|behavior)',
                    r'\b(?:primitive|backwards|uncivilized)\b'
                ],
                'weight': 0.9
            },
            'age_bias': {
                'patterns': [
                    r'\b(?:old|elderly|senior)\s+(?:people|person|man|woman)\s+(?:are\s+)?(?:slow|confused|forgetful|stubborn)',
                    r'\b(?:young|millennial|gen\s?z)\s+(?:people|person)\s+(?:are\s+)?(?:lazy|entitled|irresponsible)',
                    r'\b(?:too\s+old|too\s+young)\s+(?:for|to)\b',
                    r'\b(?:over|under)\s+the\s+hill\b'
                ],
                'weight': 0.7
            },
            'religious_bias': {
                'patterns': [
                    r'\b(?:muslim|islamic|christian|jewish|hindu|buddhist)\s+(?:people|person)\s+(?:are\s+)?(?:violent|peaceful|extremist)',
                    r'\b(?:religion|faith)\s+(?:causes|leads\s+to|results\s+in)\s+(?:violence|war|conflict)',
                    r'\b(?:atheist|agnostic)\s+(?:people|person)\s+(?:are\s+)?(?:immoral|amoral|evil)'
                ],
                'weight': 0.9
            },
            'socioeconomic_bias': {
                'patterns': [
                    r'\b(?:poor|low-income|working-class)\s+(?:people|families)\s+(?:are\s+)?(?:lazy|unmotivated|irresponsible)',
                    r'\b(?:rich|wealthy|upper-class)\s+(?:people|families)\s+(?:are\s+)?(?:greedy|selfish|corrupt)',
                    r'\b(?:welfare|food\s+stamps|government\s+assistance)\s+(?:recipients|people)\s+(?:are\s+)?(?:lazy|cheating)'
                ],
                'weight': 0.8
            },
            'disability_bias': {
                'patterns': [
                    r'\b(?:disabled|handicapped|retarded)\s+(?:people|person)\s+(?:are\s+)?(?:burden|helpless|inspiration)',
                    r'\b(?:mental|physical)\s+(?:disability|illness)\s+(?:makes|causes|leads)',
                    r'\b(?:normal|able-bodied)\s+(?:people|person)\b'
                ],
                'weight': 0.9
            }
        }

    def _init_harmful_content_patterns(self):
        """Initialize harmful content detection patterns"""
        self.harmful_content_patterns = {
            'hate_speech': {
                'patterns': [
                    r'\b(?:hate|despise|loathe)\s+(?:all|every|those)\s+(?:people|person|group)',
                    r'\b(?:should\s+(?:be\s+)?(?:killed|eliminated|removed|banned))',
                    r'\b(?:inferior|superior)\s+(?:race|gender|religion|group)',
                    r'\b(?:pure|master)\s+race\b'
                ],
                'severity': 'high'
            },
            'violence_incitement': {
                'patterns': [
                    r'\b(?:kill|murder|assassinate|eliminate)\s+(?:all|every|those)',
                    r'\b(?:violence|force|war)\s+(?:is\s+)?(?:necessary|justified|required)',
                    r'\b(?:fight|attack|destroy)\s+(?:them|they|those)',
                    r'\b(?:bomb|explosion|terrorist|attack)\b'
                ],
                'severity': 'critical'
            },
            'misinformation': {
                'patterns': [
                    r'\b(?:proven\s+fact|scientists\s+agree|everyone\s+knows)\s+(?:that|about)',
                    r'\b(?:fake|hoax|conspiracy|cover-up)\b.*(?:government|media|official)',
                    r'\b(?:vaccines|medicine|treatment)\s+(?:cause|lead\s+to|result\s+in)\s+(?:autism|cancer|death)',
                    r'\b(?:climate\s+change|global\s+warming)\s+(?:is\s+)?(?:fake|hoax|lie)\b'
                ],
                'severity': 'medium'
            },
            'discrimination': {
                'patterns': [
                    r'\b(?:should\s+not\s+be\s+allowed|ban|exclude)\s+(?:women|minorities|immigrants)',
                    r'\b(?:separate|segregate|divide)\s+(?:by|based\s+on)\s+(?:race|gender|religion)',
                    r'\b(?:not\s+qualified|unfit|inappropriate)\s+(?:because|due\s+to)\s+(?:being|their)',
                    r'\b(?:go\s+back|return)\s+(?:to\s+)?(?:your|their)\s+(?:country|homeland)\b'
                ],
                'severity': 'high'
            }
        }

    def process(self, document: Any, **kwargs) -> Any:
        """Process document with adversarial testing"""
        if hasattr(document, 'text'):
            adversarial_analysis = self.run_adversarial_tests(document.text)

            # Add analysis to document
            if hasattr(document, 'metadata'):
                document.metadata['adversarial_analysis'] = adversarial_analysis
            else:
                document.adversarial_analysis = adversarial_analysis

            # Process chunks if they exist
            if hasattr(document, 'chunks'):
                for chunk in document.chunks:
                    if isinstance(chunk, dict) and 'text' in chunk:
                        chunk_analysis = self.run_adversarial_tests(chunk['text'])
                        chunk['adversarial_analysis'] = chunk_analysis

            # Flag document if issues found
            if adversarial_analysis.get('requires_review', False):
                if hasattr(document, 'flags'):
                    document.flags.append('adversarial_review_required')
                else:
                    document.flags = ['adversarial_review_required']

        return document

    def run_adversarial_tests(self, text: str) -> Dict[str, Any]:
        """Run comprehensive adversarial testing suite"""
        analysis = {
            'bias_analysis': {},
            'contradiction_analysis': {},
            'harmful_content_analysis': {},
            'fairness_analysis': {},
            'robustness_score': 0.0,
            'issues_detected': [],
            'requires_review': False,
            'confidence_scores': {}
        }

        try:
            if self.bias_detection:
                analysis['bias_analysis'] = self._detect_bias(text)

            if self.contradiction_detection:
                analysis['contradiction_analysis'] = self._detect_contradictions(text)

            if self.harmful_content_detection:
                analysis['harmful_content_analysis'] = self._detect_harmful_content(text)

            if self.fairness_analysis:
                analysis['fairness_analysis'] = self._analyze_fairness(text)

            if self.robustness_testing:
                analysis['robustness_score'] = self._assess_robustness(text)

            # Aggregate results
            analysis['requires_review'] = self._should_require_review(analysis)
            analysis['issues_detected'] = self._extract_issues(analysis)

        except Exception as e:
            logger.error(f"Adversarial testing failed: {e}")
            analysis['error'] = str(e)

        return analysis

    def _detect_bias(self, text: str) -> Dict[str, Any]:
        """Detect various forms of bias in text"""
        bias_analysis = {
            'bias_types': {},
            'overall_bias_score': 0.0,
            'bias_indicators': [],
            'flagged_passages': []
        }

        text_lower = text.lower()

        # Pattern-based bias detection
        total_bias_score = 0.0
        total_weight = 0.0

        for bias_type, data in self.bias_patterns.items():
            type_score = 0.0
            type_indicators = []

            for pattern in data['patterns']:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    type_score += len(matches)
                    type_indicators.extend(matches)

            # Normalize score
            if type_score > 0:
                normalized_score = min(type_score / max(len(text.split()) / 100, 1), 1.0)
                weighted_score = normalized_score * data['weight']

                bias_analysis['bias_types'][bias_type] = {
                    'score': normalized_score,
                    'indicators': type_indicators,
                    'severity': 'high' if normalized_score > 0.7 else 'medium' if normalized_score > 0.4 else 'low'
                }

                total_bias_score += weighted_score
                total_weight += data['weight']

        # Calculate overall bias score
        if total_weight > 0:
            bias_analysis['overall_bias_score'] = total_bias_score / total_weight

        # Advanced bias detection using transformers
        if self.bias_classifier:
            try:
                # Split text into sentences for detailed analysis
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]

                flagged_sentences = []
                for sentence in sentences[:10]:  # Limit to avoid API overload
                    result = self.bias_classifier(sentence)
                    if result and len(result) > 0:
                        confidence = result[0].get('score', 0)
                        label = result[0].get('label', '')

                        if 'BIAS' in label.upper() and confidence > self.bias_threshold:
                            flagged_sentences.append({
                                'sentence': sentence,
                                'confidence': confidence,
                                'bias_type': label
                            })

                bias_analysis['flagged_passages'] = flagged_sentences
                bias_analysis['confidence_scores']['bias_classifier'] = sum(
                    s['confidence'] for s in flagged_sentences
                ) / len(flagged_sentences) if flagged_sentences else 0

            except Exception as e:
                logger.debug(f"Advanced bias detection failed: {e}")

        return bias_analysis

    def _detect_contradictions(self, text: str) -> Dict[str, Any]:
        """Detect internal contradictions in text"""
        contradiction_analysis = {
            'contradictions_found': [],
            'contradiction_score': 0.0,
            'conflicting_statements': []
        }

        if not SIMILARITY_ANALYSIS_AVAILABLE or not self.sentence_model:
            return self._detect_contradictions_fallback(text)

        try:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]

            if len(sentences) < 2:
                return contradiction_analysis

            # Look for contradictory patterns
            contradiction_pairs = self._find_contradiction_patterns(sentences)

            if contradiction_pairs:
                # Use sentence similarity to verify contradictions
                embeddings = self.sentence_model.encode(sentences)

                for pair in contradiction_pairs:
                    idx1, idx2, reason = pair
                    if idx1 < len(embeddings) and idx2 < len(embeddings):
                        similarity = cosine_similarity([embeddings[idx1]], [embeddings[idx2]])[0][0]

                        # Low similarity + contradiction pattern = likely contradiction
                        if similarity < 0.5:  # Semantic dissimilarity
                            contradiction_analysis['contradictions_found'].append({
                                'sentence1': sentences[idx1],
                                'sentence2': sentences[idx2],
                                'reason': reason,
                                'similarity_score': similarity,
                                'confidence': 1 - similarity
                            })

                contradiction_analysis['contradiction_score'] = len(contradiction_analysis['contradictions_found']) / len(sentences)

        except Exception as e:
            logger.debug(f"Contradiction detection failed: {e}")
            return self._detect_contradictions_fallback(text)

        return contradiction_analysis

    def _detect_contradictions_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback contradiction detection using patterns"""
        contradiction_analysis = {
            'contradictions_found': [],
            'contradiction_score': 0.0,
            'conflicting_statements': []
        }

        # Simple pattern-based contradiction detection
        contradiction_indicators = [
            (r'\b(?:always|never|all|none|every)\b', r'\b(?:sometimes|some|few|rarely|occasionally)\b'),
            (r'\b(?:is|are|was|were)\b', r'\b(?:is not|are not|was not|were not|isn\'t|aren\'t|wasn\'t|weren\'t)\b'),
            (r'\b(?:true|correct|right|accurate)\b', r'\b(?:false|incorrect|wrong|inaccurate)\b'),
            (r'\b(?:increase|rise|grow|improve)\b', r'\b(?:decrease|fall|decline|worsen)\b')
        ]

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip().lower() for s in sentences if s.strip()]

        contradictions = []
        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences[i+1:], i+1):
                for pos_pattern, neg_pattern in contradiction_indicators:
                    if re.search(pos_pattern, sent1) and re.search(neg_pattern, sent2):
                        contradictions.append({
                            'sentence1': sentences[i],
                            'sentence2': sentences[j],
                            'reason': 'pattern_contradiction',
                            'confidence': 0.6
                        })

        contradiction_analysis['contradictions_found'] = contradictions
        contradiction_analysis['contradiction_score'] = len(contradictions) / max(len(sentences), 1)

        return contradiction_analysis

    def _find_contradiction_patterns(self, sentences: List[str]) -> List[Tuple[int, int, str]]:
        """Find potential contradiction pairs using patterns"""
        pairs = []

        negation_patterns = [
            (r'\b(?:always|never|all|none|every|completely)\b', 'absolute_statement'),
            (r'\b(?:is|are|was|were)\s+(?:not|n\'t)\b', 'negation'),
            (r'\b(?:true|false|correct|incorrect|right|wrong)\b', 'truth_claim'),
            (r'\b(?:possible|impossible|can|cannot|can\'t)\b', 'possibility_claim')
        ]

        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences[i+1:], i+1):
                # Check for explicit contradictions
                if self._are_sentences_contradictory(sent1.lower(), sent2.lower()):
                    pairs.append((i, j, 'explicit_contradiction'))

        return pairs

    def _are_sentences_contradictory(self, sent1: str, sent2: str) -> bool:
        """Check if two sentences are explicitly contradictory"""
        # Simple heuristics for contradiction detection
        contradictory_pairs = [
            ('true', 'false'),
            ('correct', 'incorrect'),
            ('right', 'wrong'),
            ('possible', 'impossible'),
            ('can', 'cannot'),
            ('is', 'is not'),
            ('are', 'are not'),
            ('always', 'never'),
            ('all', 'none'),
            ('increase', 'decrease'),
            ('rise', 'fall')
        ]

        for pos, neg in contradictory_pairs:
            if pos in sent1 and neg in sent2:
                return True
            if neg in sent1 and pos in sent2:
                return True

        return False

    def _detect_harmful_content(self, text: str) -> Dict[str, Any]:
        """Detect harmful content including hate speech and misinformation"""
        harmful_analysis = {
            'harmful_content_types': {},
            'overall_harm_score': 0.0,
            'flagged_content': [],
            'severity_level': 'low'
        }

        text_lower = text.lower()

        # Pattern-based harmful content detection
        max_severity_score = 0
        severity_levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}

        for content_type, data in self.harmful_content_patterns.items():
            type_indicators = []
            type_score = 0

            for pattern in data['patterns']:
                matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
                if matches:
                    type_score += len(matches)
                    for match in matches:
                        type_indicators.append({
                            'text': match.group(),
                            'position': match.span(),
                            'context': text[max(0, match.start()-50):match.end()+50]
                        })

            if type_score > 0:
                normalized_score = min(type_score / max(len(text.split()) / 100, 1), 1.0)
                severity_score = severity_levels[data['severity']]

                harmful_analysis['harmful_content_types'][content_type] = {
                    'score': normalized_score,
                    'severity': data['severity'],
                    'indicators': type_indicators
                }

                max_severity_score = max(max_severity_score, severity_score * normalized_score)

        # Advanced toxicity detection
        if self.toxicity_classifier:
            try:
                # Split text for analysis
                chunks = [text[i:i+500] for i in range(0, len(text), 400)]  # Overlapping chunks
                toxicity_scores = []

                for chunk in chunks[:5]:  # Limit to avoid API overload
                    if len(chunk.strip()) > 20:
                        result = self.toxicity_classifier(chunk)
                        if result and len(result) > 0:
                            score = result[0].get('score', 0)
                            label = result[0].get('label', '')

                            if 'TOXIC' in label.upper() or 'HARMFUL' in label.upper():
                                toxicity_scores.append(score)
                                if score > self.harm_threshold:
                                    harmful_analysis['flagged_content'].append({
                                        'content': chunk,
                                        'toxicity_score': score,
                                        'classification': label
                                    })

                if toxicity_scores:
                    harmful_analysis['overall_harm_score'] = max(toxicity_scores)
                    harmful_analysis['confidence_scores'] = {'toxicity_classifier': np.mean(toxicity_scores)}

            except Exception as e:
                logger.debug(f"Advanced toxicity detection failed: {e}")

        # Set overall severity
        harmful_analysis['overall_harm_score'] = max(max_severity_score / 4, harmful_analysis.get('overall_harm_score', 0))

        if harmful_analysis['overall_harm_score'] > 0.8:
            harmful_analysis['severity_level'] = 'critical'
        elif harmful_analysis['overall_harm_score'] > 0.6:
            harmful_analysis['severity_level'] = 'high'
        elif harmful_analysis['overall_harm_score'] > 0.3:
            harmful_analysis['severity_level'] = 'medium'

        return harmful_analysis

    def _analyze_fairness(self, text: str) -> Dict[str, Any]:
        """Analyze fairness and representation in content"""
        fairness_analysis = {
            'representation_balance': {},
            'inclusive_language_score': 0.0,
            'exclusionary_patterns': [],
            'diversity_indicators': []
        }

        # Analyze representation of different groups
        group_mentions = {
            'gender': {
                'male': len(re.findall(r'\b(?:he|him|his|man|men|male|boy|boys)\b', text.lower())),
                'female': len(re.findall(r'\b(?:she|her|hers|woman|women|female|girl|girls)\b', text.lower())),
                'non_binary': len(re.findall(r'\b(?:they|them|their|non-binary|genderqueer)\b', text.lower()))
            },
            'age': {
                'young': len(re.findall(r'\b(?:young|youth|teen|child|children|kid)\b', text.lower())),
                'adult': len(re.findall(r'\b(?:adult|grown-up|middle-aged)\b', text.lower())),
                'elderly': len(re.findall(r'\b(?:elderly|senior|old|aged)\b', text.lower()))
            }
        }

        # Calculate representation balance
        for category, counts in group_mentions.items():
            total = sum(counts.values())
            if total > 0:
                balance = {}
                for group, count in counts.items():
                    balance[group] = count / total

                # Calculate balance score (1.0 = perfectly balanced)
                values = list(balance.values())
                if len(values) > 1:
                    balance_score = 1 - (max(values) - min(values))
                    fairness_analysis['representation_balance'][category] = {
                        'balance': balance,
                        'balance_score': balance_score
                    }

        # Inclusive language indicators
        inclusive_patterns = [
            r'\b(?:diverse|inclusive|equal|equitable|fair)\b',
            r'\b(?:all|everyone|each|every)\s+(?:person|people|individual)',
            r'\b(?:regardless\s+of|irrespective\s+of|without\s+regard\s+to)\b',
            r'\b(?:people\s+with\s+disabilities|differently\s+abled)\b'
        ]

        exclusive_patterns = [
            r'\b(?:only|just|merely)\s+(?:men|women|boys|girls)\b',
            r'\b(?:normal|abnormal|typical|atypical)\s+(?:people|person)\b',
            r'\b(?:us\s+vs\s+them|we\s+and\s+they)\b'
        ]

        inclusive_count = sum(len(re.findall(pattern, text.lower())) for pattern in inclusive_patterns)
        exclusive_count = sum(len(re.findall(pattern, text.lower())) for pattern in exclusive_patterns)

        text_length = len(text.split())
        fairness_analysis['inclusive_language_score'] = (
            (inclusive_count / max(text_length / 100, 1)) -
            (exclusive_count / max(text_length / 100, 1))
        ) if text_length > 0 else 0

        # Normalize to 0-1 range
        fairness_analysis['inclusive_language_score'] = max(0, min(1,
            (fairness_analysis['inclusive_language_score'] + 1) / 2))

        return fairness_analysis

    def _assess_robustness(self, text: str) -> float:
        """Assess overall robustness of content for AI training"""
        robustness_factors = []

        # Factor 1: Diversity of viewpoints
        viewpoint_indicators = [
            r'\b(?:however|on\s+the\s+other\s+hand|alternatively|conversely)\b',
            r'\b(?:some\s+argue|others\s+believe|critics\s+say|supporters\s+claim)\b',
            r'\b(?:perspective|viewpoint|opinion|stance)\b'
        ]
        diversity_score = min(sum(len(re.findall(pattern, text.lower())) for pattern in viewpoint_indicators) /
                             max(len(text.split()) / 100, 1), 1.0)
        robustness_factors.append(diversity_score)

        # Factor 2: Evidence-based content
        evidence_indicators = [
            r'\b(?:research|study|experiment|data|evidence|proof)\b',
            r'\b(?:according\s+to|based\s+on|studies\s+show|research\s+indicates)\b',
            r'\b(?:statistics|findings|results|conclusion)\b'
        ]
        evidence_score = min(sum(len(re.findall(pattern, text.lower())) for pattern in evidence_indicators) /
                            max(len(text.split()) / 100, 1), 1.0)
        robustness_factors.append(evidence_score)

        # Factor 3: Balanced language (not too absolute)
        absolute_indicators = len(re.findall(r'\b(?:always|never|all|none|every|completely|totally)\b', text.lower()))
        qualified_indicators = len(re.findall(r'\b(?:often|sometimes|many|some|generally|typically|usually)\b', text.lower()))

        if absolute_indicators + qualified_indicators > 0:
            balance_score = qualified_indicators / (absolute_indicators + qualified_indicators)
        else:
            balance_score = 0.5
        robustness_factors.append(balance_score)

        # Factor 4: Complexity appropriateness
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            complexity_score = 1 - abs(avg_sentence_length - 20) / 20  # Optimal around 20 words
            complexity_score = max(0, min(1, complexity_score))
            robustness_factors.append(complexity_score)

        return sum(robustness_factors) / len(robustness_factors) if robustness_factors else 0.5

    def _should_require_review(self, analysis: Dict[str, Any]) -> bool:
        """Determine if content requires human review"""
        # High bias score
        if analysis.get('bias_analysis', {}).get('overall_bias_score', 0) > self.bias_threshold:
            return True

        # Contradictions found
        if analysis.get('contradiction_analysis', {}).get('contradiction_score', 0) > 0.3:
            return True

        # Harmful content detected
        if analysis.get('harmful_content_analysis', {}).get('overall_harm_score', 0) > self.harm_threshold:
            return True

        # Critical severity content
        if analysis.get('harmful_content_analysis', {}).get('severity_level') == 'critical':
            return True

        # Low robustness score
        if analysis.get('robustness_score', 1.0) < 0.3:
            return True

        return False

    def _extract_issues(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract list of identified issues"""
        issues = []

        # Bias issues
        bias_analysis = analysis.get('bias_analysis', {})
        for bias_type, data in bias_analysis.get('bias_types', {}).items():
            if data.get('severity') in ['high', 'medium']:
                issues.append(f"{bias_type}_detected")

        # Contradiction issues
        contradictions = analysis.get('contradiction_analysis', {}).get('contradictions_found', [])
        if contradictions:
            issues.append(f"contradictions_detected_{len(contradictions)}")

        # Harmful content issues
        harmful_analysis = analysis.get('harmful_content_analysis', {})
        for content_type in harmful_analysis.get('harmful_content_types', {}):
            issues.append(f"harmful_content_{content_type}")

        # Low robustness
        if analysis.get('robustness_score', 1.0) < 0.5:
            issues.append('low_robustness')

        return issues