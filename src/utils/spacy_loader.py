"""
Unified SpaCy Model Loader with Fallback Mechanisms
Provides robust spaCy model loading with multiple fallback strategies
"""

import logging
import spacy
from typing import Optional

logger = logging.getLogger(__name__)

# Model preference order - will try each in sequence
SPACY_MODEL_PRIORITY = [
    "en_core_web_sm",     # Preferred - small, fast, covers most use cases
    "en_core_web_md",     # Medium model with vectors
    "en_core_web_lg",     # Large model with more vectors
    "en",                 # Shorthand for any English model
]

class SpaCyLoader:
    """Centralized spaCy model loader with fallback mechanisms"""

    _cached_model: Optional[spacy.Language] = None
    _model_name: Optional[str] = None

    @classmethod
    def load_model(cls, preferred_model: str = None) -> Optional[spacy.Language]:
        """
        Load a spaCy model with fallback mechanisms

        Args:
            preferred_model: Specific model name to try first

        Returns:
            spacy.Language: Loaded model or None if no models available
        """

        # Return cached model if available
        if cls._cached_model is not None:
            logger.debug(f"Using cached spaCy model: {cls._model_name}")
            return cls._cached_model

        # Build model priority list
        models_to_try = []
        if preferred_model:
            models_to_try.append(preferred_model)
        models_to_try.extend(SPACY_MODEL_PRIORITY)

        # Remove duplicates while preserving order
        models_to_try = list(dict.fromkeys(models_to_try))

        # Try loading models in priority order
        for model_name in models_to_try:
            try:
                logger.info(f"Attempting to load spaCy model: {model_name}")
                model = spacy.load(model_name)

                # Cache successful model
                cls._cached_model = model
                cls._model_name = model_name

                logger.info(f"Successfully loaded spaCy model: {model_name}")
                return model

            except OSError as e:
                logger.debug(f"Failed to load spaCy model '{model_name}': {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error loading spaCy model '{model_name}': {e}")
                continue

        # All models failed
        logger.warning(
            "No spaCy models could be loaded. Install a model with: "
            "python -m spacy download en_core_web_sm"
        )
        return None

    @classmethod
    def is_available(cls) -> bool:
        """Check if any spaCy model is available"""
        return cls.load_model() is not None

    @classmethod
    def get_model_name(cls) -> Optional[str]:
        """Get the name of the currently loaded model"""
        if cls._cached_model is None:
            cls.load_model()  # Try to load if not already loaded
        return cls._model_name

    @classmethod
    def clear_cache(cls):
        """Clear the cached model (for testing or reloading)"""
        cls._cached_model = None
        cls._model_name = None


def load_spacy_model(preferred_model: str = None) -> Optional[spacy.Language]:
    """
    Convenience function to load a spaCy model with fallbacks

    Args:
        preferred_model: Specific model name to try first

    Returns:
        spacy.Language: Loaded model or None if no models available
    """
    return SpaCyLoader.load_model(preferred_model)


def is_spacy_available() -> bool:
    """Check if any spaCy model is available"""
    return SpaCyLoader.is_available()