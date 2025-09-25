#!/usr/bin/env python3
"""
Unified Model Setup Script for Dataseter
Installs all required AI models with fallback mechanisms
"""

import sys
import subprocess
import logging
from pathlib import Path
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_package_availability(package_name: str) -> bool:
    """Check if a Python package is available"""
    return importlib.util.find_spec(package_name) is not None

def run_command(command: str, description: str, allow_failure: bool = True) -> bool:
    """Run a command and return success status"""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            logger.info(f"✓ Success: {description}")
            if result.stdout:
                logger.debug(f"Output: {result.stdout.strip()}")
            return True
        else:
            logger.warning(f"✗ Failed: {description}")
            if result.stderr:
                logger.warning(f"Error: {result.stderr.strip()}")
            if not allow_failure:
                sys.exit(1)
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"✗ Timeout: {description}")
        if not allow_failure:
            sys.exit(1)
        return False
    except Exception as e:
        logger.error(f"✗ Exception during {description}: {e}")
        if not allow_failure:
            sys.exit(1)
        return False

def setup_spacy_models():
    """Setup spaCy models with fallback options"""
    logger.info("=== Setting up spaCy models ===")

    if not check_package_availability('spacy'):
        logger.warning("spaCy not available, skipping model installation")
        return False

    # Try to install models in order of preference
    spacy_models = [
        "en_core_web_sm",   # Small, fast, most compatible
        "en_core_web_md",   # Medium with word vectors
        "en_core_web_lg",   # Large with more vectors
        "en",               # Generic English model
    ]

    success = False
    for model in spacy_models:
        if run_command(f"python -m spacy download {model}", f"Installing spaCy model: {model}"):
            success = True
            logger.info(f"✓ Successfully installed spaCy model: {model}")
            break
        else:
            logger.warning(f"Failed to install spaCy model: {model}, trying next...")

    if not success:
        logger.warning("No spaCy models could be installed - fallback methods will be used")
        return False

    return True

def setup_nltk_data():
    """Setup NLTK data"""
    logger.info("=== Setting up NLTK data ===")

    if not check_package_availability('nltk'):
        logger.warning("NLTK not available, skipping data download")
        return False

    nltk_datasets = [
        "punkt",
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger",
        "vader_lexicon"
    ]

    success = True
    for dataset in nltk_datasets:
        cmd = f'python -c "import nltk; nltk.download(\'{dataset}\')"'
        if not run_command(cmd, f"Downloading NLTK dataset: {dataset}"):
            success = False

    return success

def setup_transformers_cache():
    """Setup transformers models cache"""
    logger.info("=== Setting up Transformers models ===")

    if not check_package_availability('transformers'):
        logger.warning("Transformers not available, skipping model pre-download")
        return False

    # Pre-download commonly used models to avoid runtime delays
    models_to_cache = [
        "all-MiniLM-L6-v2",  # Sentence embeddings
        "dbmdz/bert-large-cased-finetuned-conll03-english",  # NER
        "cardiffnlp/twitter-roberta-base-sentiment-latest",  # Sentiment
    ]

    success = True
    for model in models_to_cache:
        # Pre-load sentence transformer
        if "MiniLM" in model:
            cmd = f'python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer(\'{model}\')"'
        else:
            cmd = f'python -c "from transformers import AutoModel; AutoModel.from_pretrained(\'{model}\')"'

        if not run_command(cmd, f"Caching transformer model: {model}"):
            success = False
            logger.warning(f"Failed to cache model {model}, will download at runtime")

    return success

def verify_installation():
    """Verify that models are properly installed"""
    logger.info("=== Verifying installation ===")

    # Test spaCy model loading using our unified loader
    spacy_test = '''
try:
    import sys
    sys.path.append('src')
    from src.utils.spacy_loader import load_spacy_model, is_spacy_available

    if is_spacy_available():
        model = load_spacy_model()
        if model:
            print("✓ spaCy model successfully loaded")
        else:
            print("✗ spaCy model loading failed")
    else:
        print("✗ No spaCy models available")
except Exception as e:
    print(f"✗ spaCy test failed: {e}")
'''

    run_command(f'python -c "{spacy_test.strip()}"', "Testing spaCy model loading")

    # Test NLTK
    nltk_test = '''
try:
    import nltk
    nltk.word_tokenize("Test sentence")
    print("✓ NLTK working correctly")
except Exception as e:
    print(f"✗ NLTK test failed: {e}")
'''

    run_command(f'python -c "{nltk_test.strip()}"', "Testing NLTK")

    # Test sentence transformers
    st_test = '''
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(["Test sentence"])
    print("✓ Sentence Transformers working correctly")
except Exception as e:
    print(f"✗ Sentence Transformers test failed: {e}")
'''

    run_command(f'python -c "{st_test.strip()}"', "Testing Sentence Transformers")

def main():
    """Main setup function"""
    logger.info("🚀 Starting Dataseter Model Setup")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Working directory: {Path.cwd()}")

    results = {}

    # Setup each component
    results['spacy'] = setup_spacy_models()
    results['nltk'] = setup_nltk_data()
    results['transformers'] = setup_transformers_cache()

    # Verify installation
    verify_installation()

    # Summary
    logger.info("=== Setup Summary ===")
    for component, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"{component.upper()}: {status}")

    if all(results.values()):
        logger.info("🎉 All models setup successfully!")
        return 0
    else:
        logger.warning("⚠️  Some models failed to install but Dataseter will use fallback methods")
        return 0  # Don't fail completely, as fallbacks are available

if __name__ == "__main__":
    sys.exit(main())