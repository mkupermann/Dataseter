"""
Knowledge Graph Extraction and Entity Relationship Mapping
Extracts structured knowledge from text for enhanced AI training datasets
"""

import re
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
import logging

try:
    import spacy
    from spacy.matcher import Matcher, DependencyMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class KnowledgeExtractor:
    """Extract structured knowledge graphs from text"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.extract_entities = config.get('extract_entities', True)
        self.extract_relations = config.get('extract_relations', True)
        self.extract_concepts = config.get('extract_concepts', True)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)

        self._init_models()
        self._init_patterns()

    def _init_models(self):
        """Initialize NLP models for knowledge extraction"""
        self.nlp = None
        self.ner_pipeline = None
        self.relation_pipeline = None

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.matcher = Matcher(self.nlp.vocab)
                self.dep_matcher = DependencyMatcher(self.nlp.vocab)
                logger.info("SpaCy models loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load SpaCy models: {e}")

        if TRANSFORMERS_AVAILABLE:
            try:
                # NER pipeline for entity extraction
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple"
                )

                # Relation extraction pipeline (if available)
                try:
                    self.relation_pipeline = pipeline(
                        "text-classification",
                        model="microsoft/DialoGPT-medium"  # Placeholder - would use actual relation extraction model
                    )
                except:
                    self.relation_pipeline = None

                logger.info("Transformer models loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load transformer models: {e}")

    def _init_patterns(self):
        """Initialize pattern matching for relations"""
        if not self.nlp:
            return

        # Define patterns for common relations
        relation_patterns = [
            # Causal relations
            ([{"LEMMA": {"IN": ["cause", "lead", "result", "due"]}}, {"POS": "ADP", "OP": "?"}, {"POS": "DET", "OP": "?"}, {"ENT_TYPE": {"IN": ["PERSON", "ORG", "EVENT"]}}],
             "CAUSES"),

            # Part-of relations
            ([{"ENT_TYPE": {"IN": ["PERSON", "ORG"]}}, {"LEMMA": {"IN": ["part", "member", "component"]}, "POS": "NOUN"}, {"LEMMA": "of"}, {"ENT_TYPE": {"IN": ["ORG", "GPE"]}}],
             "PART_OF"),

            # Location relations
            ([{"ENT_TYPE": "PERSON"}, {"LEMMA": {"IN": ["in", "at", "from", "live", "born"]}}, {"ENT_TYPE": {"IN": ["GPE", "LOC"]}}],
             "LOCATED_IN"),

            # Temporal relations
            ([{"ENT_TYPE": {"IN": ["EVENT", "ORG"]}}, {"LEMMA": {"IN": ["during", "before", "after", "since"]}}, {"ENT_TYPE": "DATE"}],
             "TEMPORAL"),

            # Professional relations
            ([{"ENT_TYPE": "PERSON"}, {"LEMMA": {"IN": ["work", "employ", "CEO", "president", "director"]}}, {"ENT_TYPE": "ORG"}],
             "WORKS_FOR"),
        ]

        # Add patterns to matcher
        for i, (pattern, label) in enumerate(relation_patterns):
            self.matcher.add(f"RELATION_{label}_{i}", [pattern])

        # Dependency patterns for more complex relations
        dependency_patterns = [
            # Subject-Verb-Object patterns
            ([
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": "nsubj"}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": "dobj"}},
            ], "SVO_RELATION"),
        ]

        for i, (pattern, label) in enumerate(dependency_patterns):
            self.dep_matcher.add(f"DEP_{label}_{i}", [pattern])

    def process(self, document: Any, **kwargs) -> Any:
        """Process document and extract knowledge graph"""
        if hasattr(document, 'text'):
            knowledge_graph = self.extract_knowledge_graph(document.text)

            # Add to document
            if hasattr(document, 'metadata'):
                document.metadata['knowledge_graph'] = knowledge_graph
            else:
                document.knowledge_graph = knowledge_graph

            # Also add to chunks if they exist
            if hasattr(document, 'chunks'):
                for chunk in document.chunks:
                    if isinstance(chunk, dict) and 'text' in chunk:
                        chunk_kg = self.extract_knowledge_graph(chunk['text'])
                        chunk['knowledge_graph'] = chunk_kg

        return document

    def extract_knowledge_graph(self, text: str) -> Dict[str, Any]:
        """Extract complete knowledge graph from text"""
        knowledge_graph = {
            'entities': [],
            'relations': [],
            'concepts': [],
            'facts': [],
            'confidence_score': 0.0
        }

        try:
            if self.extract_entities:
                knowledge_graph['entities'] = self._extract_entities(text)

            if self.extract_relations:
                knowledge_graph['relations'] = self._extract_relations(text, knowledge_graph['entities'])

            if self.extract_concepts:
                knowledge_graph['concepts'] = self._extract_concepts(text)

            # Extract factual statements
            knowledge_graph['facts'] = self._extract_facts(text, knowledge_graph['entities'], knowledge_graph['relations'])

            # Calculate overall confidence
            knowledge_graph['confidence_score'] = self._calculate_kg_confidence(knowledge_graph)

            logger.debug(f"Extracted KG: {len(knowledge_graph['entities'])} entities, {len(knowledge_graph['relations'])} relations")

        except Exception as e:
            logger.error(f"Knowledge graph extraction failed: {e}")

        return knowledge_graph

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities with additional metadata"""
        entities = []

        # Use transformer-based NER if available
        if self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(text[:1000])  # Limit for API
                for ent in ner_results:
                    if ent['score'] >= self.confidence_threshold:
                        entities.append({
                            'text': ent['word'].replace('##', ''),  # Clean subword tokens
                            'label': ent['entity_group'],
                            'confidence': ent['score'],
                            'start': ent.get('start', 0),
                            'end': ent.get('end', len(ent['word'])),
                            'source': 'transformer'
                        })
            except Exception as e:
                logger.debug(f"Transformer NER failed: {e}")

        # Use spaCy NER as backup/supplement
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'confidence': 0.8,  # Default confidence for spaCy
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'description': spacy.explain(ent.label_) if spacy.explain(ent.label_) else '',
                        'source': 'spacy'
                    })
            except Exception as e:
                logger.debug(f"SpaCy NER failed: {e}")

        # Deduplicate entities
        entities = self._deduplicate_entities(entities)

        # Add entity types and properties
        for entity in entities:
            entity.update(self._analyze_entity(entity['text'], entity['label']))

        return entities

    def _extract_relations(self, text: str, entities: List[Dict]) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        relations = []

        if not self.nlp:
            return self._extract_simple_relations(text, entities)

        try:
            doc = self.nlp(text)

            # Pattern-based relation extraction
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                relation_type = self.nlp.vocab.strings[match_id].split('_')[1]

                relations.append({
                    'type': relation_type,
                    'text': span.text,
                    'confidence': 0.7,
                    'method': 'pattern_matching',
                    'span': (start, end)
                })

            # Dependency-based relation extraction
            dep_matches = self.dep_matcher(doc)
            for match_id, token_ids in dep_matches:
                tokens = [doc[token_id] for token_id in token_ids]

                if len(tokens) >= 3:  # SVO pattern
                    subject = self._find_entity_for_token(tokens[1], entities)
                    verb = tokens[0]
                    obj = self._find_entity_for_token(tokens[2], entities)

                    if subject and obj:
                        relations.append({
                            'subject': subject,
                            'predicate': verb.lemma_,
                            'object': obj,
                            'confidence': 0.8,
                            'method': 'dependency_parsing',
                            'sentence': verb.sent.text
                        })

            # Co-occurrence based relations
            cooccurrence_relations = self._extract_cooccurrence_relations(entities, text)
            relations.extend(cooccurrence_relations)

        except Exception as e:
            logger.debug(f"Advanced relation extraction failed: {e}")
            relations = self._extract_simple_relations(text, entities)

        return relations

    def _extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract conceptual information and topics"""
        concepts = []

        try:
            if self.nlp:
                doc = self.nlp(text)

                # Extract noun phrases as concepts
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) > 1:  # Multi-word concepts
                        concepts.append({
                            'text': chunk.text,
                            'type': 'noun_phrase',
                            'root': chunk.root.lemma_,
                            'confidence': 0.6,
                            'pos_tags': [token.pos_ for token in chunk]
                        })

                # Extract verb phrases that indicate actions/processes
                for token in doc:
                    if token.pos_ == "VERB" and token.dep_ == "ROOT":
                        verb_phrase = self._extract_verb_phrase(token)
                        if verb_phrase:
                            concepts.append({
                                'text': verb_phrase,
                                'type': 'action',
                                'root': token.lemma_,
                                'confidence': 0.7
                            })

            # Domain-specific concept extraction
            domain_concepts = self._extract_domain_concepts(text)
            concepts.extend(domain_concepts)

        except Exception as e:
            logger.debug(f"Concept extraction failed: {e}")

        return concepts

    def _extract_facts(self, text: str, entities: List[Dict], relations: List[Dict]) -> List[Dict[str, Any]]:
        """Extract factual statements from text"""
        facts = []

        try:
            # Extract statements with high factual confidence
            fact_patterns = [
                r'(?:is|are|was|were|has|have)\s+(.+?)(?:\.|,|;)',
                r'(?:according to|research shows|studies indicate|data suggests)\s+(.+?)(?:\.|,|;)',
                r'(?:in \d{4}|on [A-Z][a-z]+ \d+)\s*[,:]?\s*(.+?)(?:\.|,|;)'
            ]

            sentences = re.split(r'[.!?]+', text)

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:  # Skip very short sentences
                    continue

                # Check if sentence matches factual patterns
                factual_confidence = self._calculate_factual_confidence(sentence)

                if factual_confidence > 0.5:
                    # Find related entities and relations in this sentence
                    sentence_entities = [e for e in entities
                                       if e['text'].lower() in sentence.lower()]
                    sentence_relations = [r for r in relations
                                        if isinstance(r, dict) and
                                        r.get('sentence', '').lower() == sentence.lower()]

                    facts.append({
                        'statement': sentence,
                        'confidence': factual_confidence,
                        'entities': sentence_entities,
                        'relations': sentence_relations,
                        'type': self._classify_fact_type(sentence)
                    })

        except Exception as e:
            logger.debug(f"Fact extraction failed: {e}")

        return facts

    def _calculate_kg_confidence(self, kg: Dict) -> float:
        """Calculate overall confidence score for knowledge graph"""
        scores = []

        # Entity confidence
        if kg['entities']:
            entity_scores = [e.get('confidence', 0.5) for e in kg['entities']]
            scores.append(sum(entity_scores) / len(entity_scores))

        # Relation confidence
        if kg['relations']:
            relation_scores = [r.get('confidence', 0.5) for r in kg['relations'] if isinstance(r, dict)]
            if relation_scores:
                scores.append(sum(relation_scores) / len(relation_scores))

        # Fact confidence
        if kg['facts']:
            fact_scores = [f.get('confidence', 0.5) for f in kg['facts']]
            scores.append(sum(fact_scores) / len(fact_scores))

        return sum(scores) / len(scores) if scores else 0.5

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities"""
        seen = set()
        unique_entities = []

        for entity in entities:
            key = (entity['text'].lower(), entity['label'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
            else:
                # Merge confidence scores
                for existing in unique_entities:
                    if (existing['text'].lower(), existing['label']) == key:
                        existing['confidence'] = max(existing['confidence'], entity['confidence'])
                        break

        return unique_entities

    def _analyze_entity(self, text: str, label: str) -> Dict[str, Any]:
        """Add additional analysis for entities"""
        analysis = {
            'canonical_form': text.title() if label in ['PERSON', 'ORG', 'GPE'] else text,
            'entity_type': self._get_detailed_entity_type(text, label),
            'properties': self._extract_entity_properties(text, label)
        }

        return analysis

    def _get_detailed_entity_type(self, text: str, label: str) -> str:
        """Get more detailed entity type"""
        type_mapping = {
            'PERSON': 'person',
            'ORG': 'organization',
            'GPE': 'location',
            'LOC': 'location',
            'DATE': 'temporal',
            'TIME': 'temporal',
            'MONEY': 'monetary',
            'PERCENT': 'percentage',
            'CARDINAL': 'number',
            'ORDINAL': 'number'
        }

        return type_mapping.get(label, 'other')

    def _extract_entity_properties(self, text: str, label: str) -> List[str]:
        """Extract properties of entities"""
        properties = []

        # Add label-specific properties
        if label == 'PERSON':
            if any(title in text.lower() for title in ['dr', 'prof', 'president', 'ceo']):
                properties.append('has_title')
        elif label == 'ORG':
            if any(word in text.lower() for word in ['company', 'corp', 'inc', 'ltd']):
                properties.append('company')
            elif any(word in text.lower() for word in ['university', 'college', 'school']):
                properties.append('educational')

        return properties

    def _extract_simple_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Simple pattern-based relation extraction"""
        relations = []

        # Simple co-occurrence relations
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence_entities = [e for e in entities if e['text'] in sentence]

            # Find entity pairs in same sentence
            for i, ent1 in enumerate(sentence_entities):
                for ent2 in sentence_entities[i+1:]:
                    relations.append({
                        'subject': ent1['text'],
                        'object': ent2['text'],
                        'predicate': 'co_occurs_with',
                        'confidence': 0.3,
                        'method': 'co_occurrence',
                        'context': sentence
                    })

        return relations

    def _find_entity_for_token(self, token, entities: List[Dict]) -> Optional[str]:
        """Find entity that contains this token"""
        for entity in entities:
            if token.text.lower() in entity['text'].lower():
                return entity['text']
        return None

    def _extract_verb_phrase(self, verb_token) -> Optional[str]:
        """Extract verb phrase around a verb token"""
        if not verb_token:
            return None

        # Simple verb phrase extraction
        phrase_tokens = [verb_token]

        # Add auxiliary verbs
        for child in verb_token.children:
            if child.dep_ in ['aux', 'auxpass']:
                phrase_tokens.append(child)

        # Add particles
        for child in verb_token.children:
            if child.dep_ == 'prt':
                phrase_tokens.append(child)

        phrase_tokens.sort(key=lambda x: x.i)
        return ' '.join(token.text for token in phrase_tokens)

    def _extract_cooccurrence_relations(self, entities: List[Dict], text: str) -> List[Dict]:
        """Extract relations based on entity co-occurrence"""
        relations = []
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence_entities = [e for e in entities if e['text'] in sentence]

            if len(sentence_entities) >= 2:
                for i, ent1 in enumerate(sentence_entities):
                    for ent2 in sentence_entities[i+1:]:
                        relations.append({
                            'subject': ent1['text'],
                            'object': ent2['text'],
                            'predicate': 'mentioned_with',
                            'confidence': 0.4,
                            'method': 'co_occurrence',
                            'context': sentence.strip()
                        })

        return relations

    def _extract_domain_concepts(self, text: str) -> List[Dict]:
        """Extract domain-specific concepts"""
        concepts = []

        # Technical/scientific concepts
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+-\w+(?:-\w+)*\b',  # Hyphenated terms
            r'\b\w+(?:ology|ometry|ics|tion|sion)\b',  # Technical suffixes
        ]

        for pattern in technical_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                term = match.group()
                if len(term) > 3:  # Filter very short terms
                    concepts.append({
                        'text': term,
                        'type': 'technical_term',
                        'confidence': 0.5,
                        'pattern': pattern
                    })

        return concepts

    def _calculate_factual_confidence(self, sentence: str) -> float:
        """Calculate confidence that a sentence contains factual information"""
        confidence = 0.3  # Base confidence

        # Positive indicators
        factual_indicators = [
            r'\b(?:is|are|was|were)\b',  # State verbs
            r'\b(?:according to|research|study|data)\b',  # Authority
            r'\b\d{4}\b',  # Years
            r'\b(?:percent|%|million|billion)\b',  # Numbers
        ]

        # Negative indicators
        opinion_indicators = [
            r'\b(?:think|believe|feel|seem|appear)\b',
            r'\b(?:maybe|perhaps|possibly|probably)\b',
            r'\b(?:I|we|you)\b',  # Personal pronouns
        ]

        for pattern in factual_indicators:
            if re.search(pattern, sentence, re.IGNORECASE):
                confidence += 0.2

        for pattern in opinion_indicators:
            if re.search(pattern, sentence, re.IGNORECASE):
                confidence -= 0.3

        return max(0.0, min(1.0, confidence))

    def _classify_fact_type(self, statement: str) -> str:
        """Classify the type of factual statement"""
        statement_lower = statement.lower()

        if any(word in statement_lower for word in ['define', 'definition', 'means', 'refers to']):
            return 'definition'
        elif any(word in statement_lower for word in ['cause', 'result', 'due to', 'because']):
            return 'causal'
        elif any(word in statement_lower for word in ['research', 'study', 'experiment']):
            return 'empirical'
        elif re.search(r'\b\d{4}\b', statement):
            return 'temporal'
        elif any(word in statement_lower for word in ['percent', '%', 'million', 'billion']):
            return 'quantitative'
        else:
            return 'descriptive'