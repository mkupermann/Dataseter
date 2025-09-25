"""
Privacy protection and PII detection/redaction
"""

import re
import logging
import hashlib
from typing import Dict, Any, List

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class PrivacyProtector:
    """Detect and redact PII from text"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.detect_pii = config.get('detect_pii', True)
        self.pii_entities = config.get('pii_entities', 
            ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'US_SSN', 'CREDIT_CARD', 'IP_ADDRESS'])
        self.redaction_method = config.get('redaction_method', 'mask')  # mask, remove, hash, encrypt
        self.custom_patterns = config.get('custom_patterns', [])
        
        if PRESIDIO_AVAILABLE:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        else:
            logger.warning("Presidio not available, using basic regex patterns")

    def process(self, document: Any, **kwargs) -> Any:
        """Process document to remove PII"""
        if not self.detect_pii:
            return document
            
        if hasattr(document, 'text'):
            document.text = self.protect_text(document.text)
        if hasattr(document, 'chunks'):
            for chunk in document.chunks:
                if 'text' in chunk:
                    chunk['text'] = self.protect_text(chunk['text'])
        return document

    def protect_text(self, text: str) -> str:
        """Detect and redact PII from text"""
        if PRESIDIO_AVAILABLE:
            return self._protect_with_presidio(text)
        else:
            return self._protect_with_regex(text)

    def _protect_with_presidio(self, text: str) -> str:
        """Use Presidio for PII detection"""
        results = self.analyzer.analyze(text, entities=self.pii_entities, language='en')
        
        if self.redaction_method == 'mask':
            anonymized = self.anonymizer.anonymize(text, results)
            return anonymized.text
        elif self.redaction_method == 'hash':
            for result in reversed(results):
                entity = text[result.start:result.end]
                hashed = hashlib.sha256(entity.encode()).hexdigest()[:8]
                text = text[:result.start] + f"[{result.entity_type}_{hashed}]" + text[result.end:]
            return text
        else:
            for result in reversed(results):
                text = text[:result.start] + f"[{result.entity_type}]" + text[result.end:]
            return text

    def _protect_with_regex(self, text: str) -> str:
        """Use regex patterns for basic PII detection"""
        patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
            'CREDIT_CARD': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'IP_ADDRESS': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        }
        
        for entity_type, pattern in patterns.items():
            if self.redaction_method == 'mask':
                text = re.sub(pattern, f'[{entity_type}]', text)
            elif self.redaction_method == 'remove':
                text = re.sub(pattern, '', text)
                
        # Apply custom patterns
        for pattern_info in self.custom_patterns:
            pattern = pattern_info.get('pattern')
            replacement = pattern_info.get('replacement', '[REDACTED]')
            text = re.sub(pattern, replacement, text)
            
        return text