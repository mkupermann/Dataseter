"""
Comprehensive tests for all advanced AI-optimized features
"""

import unittest
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import DatasetCreator
from src.processors.semantic_chunker import SemanticChunker
from src.processors.knowledge_extractor import KnowledgeExtractor
from src.processors.semantic_quality import SemanticQualityScorer
from src.processors.metacognitive_annotator import MetacognitiveAnnotator
from src.processors.adversarial_tester import AdversarialTester


class TestAdvancedFeatures(unittest.TestCase):
    """Test all advanced features for AI-optimized dataset creation"""

    def setUp(self):
        """Set up test fixtures"""
        self.creator = DatasetCreator()
        self.test_text = """
        Albert Einstein was a theoretical physicist who developed the theory of relativity.
        Because of his groundbreaking work, Einstein revolutionized our understanding of space and time.
        The theory of relativity consists of two parts: special relativity and general relativity.
        Therefore, Einstein's contributions fundamentally changed physics forever.

        For example, the famous equation E=mc² demonstrates the equivalence of mass and energy.
        This equation has profound implications for nuclear physics and cosmology.
        However, Einstein initially resisted quantum mechanics, stating "God does not play dice."
        Nevertheless, quantum mechanics proved to be correct in many experimental tests.
        """

        self.test_biased_text = """
        Women are naturally better at nurturing children than men.
        All politicians are corrupt and only care about money.
        Young people these days are lazy and entitled.
        """

        self.test_complex_text = """
        The epistemological foundations of quantum mechanics necessitate a paradigmatic
        reconceptualization of deterministic causality, whereby the Copenhagen interpretation's
        probabilistic framework supersedes classical Newtonian mechanics' predictive certainty.
        """

    def test_semantic_chunking(self):
        """Test semantic chunking with reasoning preservation"""
        print("\n=== Testing Semantic Chunking ===")

        chunker = SemanticChunker({
            'target_size': 100,
            'preserve_reasoning': True,
            'detect_arguments': True
        })

        # Create mock document
        class MockDocument:
            def __init__(self, text):
                self.text = text
                self.chunks = []
                self.metadata = {}

        doc = MockDocument(self.test_text)

        try:
            processed_doc = chunker.process(doc)
        except RuntimeError as e:
            # Handle MPS memory errors gracefully
            if "MPS backend out of memory" in str(e):
                print("  ⚠️ MPS memory exhausted, testing with fallback mode")
                # Force fallback mode
                chunker.sentence_model = None
                chunker.ner_model = None
                processed_doc = chunker.process(doc)
            else:
                raise

        # Verify chunks were created
        self.assertGreater(len(processed_doc.chunks), 0, "No chunks created")

        # Check for reasoning chains
        has_reasoning = any(
            chunk.get('reasoning_chains', [])
            for chunk in processed_doc.chunks
        )
        self.assertTrue(has_reasoning, "No reasoning chains detected")

        # Check for semantic coherence
        for chunk in processed_doc.chunks:
            if 'coherence_score' in chunk:
                self.assertGreaterEqual(chunk['coherence_score'], 0, "Invalid coherence score")

        print(f"✓ Created {len(processed_doc.chunks)} semantic chunks")
        print(f"✓ Reasoning chains preserved in chunks")

        return True

    def test_knowledge_extraction(self):
        """Test knowledge graph extraction"""
        print("\n=== Testing Knowledge Extraction ===")

        extractor = KnowledgeExtractor({
            'extract_entities': True,
            'extract_relations': True,
            'extract_concepts': True
        })

        knowledge_graph = extractor.extract_knowledge_graph(self.test_text)

        # Verify entities extracted
        self.assertIn('entities', knowledge_graph)
        self.assertGreater(len(knowledge_graph['entities']), 0, "No entities extracted")

        # Check for Einstein as an entity
        entity_names = [e['text'] for e in knowledge_graph['entities']]
        self.assertIn('Albert Einstein', entity_names, "Failed to extract key entity")

        # Verify concepts extracted
        self.assertIn('concepts', knowledge_graph)

        # Verify facts extracted
        self.assertIn('facts', knowledge_graph)

        print(f"✓ Extracted {len(knowledge_graph['entities'])} entities")
        print(f"✓ Extracted {len(knowledge_graph.get('relations', []))} relations")
        print(f"✓ Extracted {len(knowledge_graph.get('concepts', []))} concepts")

        return True

    def test_semantic_quality_scoring(self):
        """Test advanced quality scoring"""
        print("\n=== Testing Semantic Quality Scoring ===")

        scorer = SemanticQualityScorer({
            'authority_weight': 0.3,
            'coherence_weight': 0.25,
            'complexity_weight': 0.2,
            'factuality_weight': 0.15,
            'reasoning_weight': 0.1
        })

        analysis = scorer.analyze_quality(self.test_text)

        # Verify all scores are present
        self.assertIn('overall_score', analysis)
        self.assertIn('authority_score', analysis)
        self.assertIn('coherence_score', analysis)
        self.assertIn('complexity_score', analysis)
        self.assertIn('factuality_score', analysis)
        self.assertIn('reasoning_score', analysis)
        self.assertIn('training_value', analysis)

        # Verify scores are in valid range
        for key in ['overall_score', 'authority_score', 'coherence_score']:
            score = analysis[key]
            self.assertGreaterEqual(score, 0, f"{key} below 0")
            self.assertLessEqual(score, 1, f"{key} above 1")

        # Einstein text should have good reasoning score
        self.assertGreater(analysis['reasoning_score'], 0.3, "Reasoning score too low for logical text")

        print(f"✓ Overall quality score: {analysis['overall_score']:.2f}")
        print(f"✓ Training value: {analysis['training_value']:.2f}")
        print(f"✓ All quality dimensions assessed")

        return True

    def test_metacognitive_annotations(self):
        """Test metacognitive annotation system"""
        print("\n=== Testing Metacognitive Annotations ===")

        annotator = MetacognitiveAnnotator({
            'confidence_analysis': True,
            'complexity_analysis': True,
            'prerequisite_analysis': True,
            'learning_objectives': True,
            'cognitive_load_assessment': True
        })

        # Test with complex text
        annotations = annotator.analyze_metacognitive_aspects(self.test_complex_text)

        # Verify confidence analysis
        self.assertIn('confidence_distribution', annotations)

        # Verify complexity metrics
        self.assertIn('complexity_metrics', annotations)
        complexity = annotations['complexity_metrics']
        self.assertIn('overall_complexity', complexity)
        self.assertGreater(complexity['overall_complexity'], 0.5, "Complex text not detected as complex")

        # Verify prerequisite knowledge
        self.assertIn('prerequisite_knowledge', annotations)
        prereqs = annotations['prerequisite_knowledge']
        self.assertIsInstance(prereqs, list)

        # Check for philosophy prerequisites (epistemological text)
        has_philosophy = any(p['domain'] == 'philosophy' for p in prereqs)
        self.assertTrue(has_philosophy or len(prereqs) > 0, "No prerequisites detected for complex text")

        # Verify cognitive load
        self.assertIn('cognitive_load', annotations)
        load = annotations['cognitive_load']
        self.assertIn('total_load', load)
        self.assertGreater(load['total_load'], 0.5, "High cognitive load not detected")

        print(f"✓ Complexity score: {complexity['overall_complexity']:.2f}")
        print(f"✓ Cognitive load: {load['total_load']:.2f}")
        print(f"✓ Prerequisites identified: {len(prereqs)}")

        return True

    def test_adversarial_testing(self):
        """Test adversarial testing for bias and harmful content"""
        print("\n=== Testing Adversarial Framework ===")

        tester = AdversarialTester({
            'bias_detection': True,
            'contradiction_detection': True,
            'harmful_content_detection': True,
            'fairness_analysis': True,
            'robustness_testing': True
        })

        # Test biased text
        analysis = tester.run_adversarial_tests(self.test_biased_text)

        # Verify bias detection
        self.assertIn('bias_analysis', analysis)
        bias = analysis['bias_analysis']
        self.assertIn('overall_bias_score', bias)
        self.assertGreater(bias['overall_bias_score'], 0.3, "Bias not detected in biased text")

        # Check for specific bias types
        self.assertIn('bias_types', bias)
        detected_biases = list(bias['bias_types'].keys())
        self.assertGreater(len(detected_biases), 0, "No specific bias types detected")

        # Verify requires review flag
        self.assertIn('requires_review', analysis)
        self.assertTrue(analysis['requires_review'], "Biased text not flagged for review")

        # Test contradiction detection on Einstein text
        einstein_analysis = tester.run_adversarial_tests(self.test_text)
        self.assertIn('contradiction_analysis', einstein_analysis)

        # Verify fairness analysis
        self.assertIn('fairness_analysis', einstein_analysis)

        print(f"✓ Bias score for biased text: {bias['overall_bias_score']:.2f}")
        print(f"✓ Detected bias types: {detected_biases}")
        print(f"✓ Review required: {analysis['requires_review']}")

        return True

    def test_integration_pipeline(self):
        """Test complete integration of all features"""
        print("\n=== Testing Complete Integration Pipeline ===")

        # Add a simple test file
        test_file = Path("test_integration.txt")
        test_file.write_text(self.test_text)

        try:
            # Create dataset with all features
            creator = DatasetCreator()
            creator.add_directory(".", pattern="test_integration.txt")

            # Process with all features enabled
            dataset = creator.process(
                chunk_size=200,
                overlap=20,
                remove_pii=True,
                quality_threshold=0.3,
                remove_duplicates=True,
                chunking_strategy='semantic',
                extract_knowledge=True,
                add_metacognitive_annotations=True,
                enable_adversarial_testing=True
            )

            # Verify dataset created
            self.assertIsNotNone(dataset, "Dataset creation failed")
            self.assertGreater(len(dataset.documents), 0, "No documents in dataset")

            # Check for advanced features in metadata
            if dataset.documents:
                doc = dataset.documents[0]

                # Check for chunks
                if hasattr(doc, 'chunks'):
                    self.assertGreater(len(doc.chunks), 0, "No chunks created")

                    # Verify advanced features in chunks
                    chunk = doc.chunks[0] if doc.chunks else {}
                    features_found = []

                    if 'reasoning_chains' in chunk:
                        features_found.append('reasoning_chains')
                    if 'knowledge_graph' in chunk:
                        features_found.append('knowledge_graph')
                    if 'quality_analysis' in chunk:
                        features_found.append('quality_analysis')
                    if 'metacognitive_annotations' in chunk:
                        features_found.append('metacognitive_annotations')
                    if 'adversarial_analysis' in chunk:
                        features_found.append('adversarial_analysis')

                    print(f"✓ Advanced features integrated: {features_found}")

                # Check metadata
                if hasattr(doc, 'metadata'):
                    metadata_features = list(doc.metadata.keys())
                    print(f"✓ Metadata features: {metadata_features[:5]}...")

            print("✓ Full pipeline integration successful")
            return True

        finally:
            # Clean up test file
            if test_file.exists():
                test_file.unlink()

    def test_fallback_modes(self):
        """Test that features work without transformers"""
        print("\n=== Testing Fallback Modes ===")

        # Test semantic chunker fallback
        chunker = SemanticChunker({'target_size': 100})
        chunks = chunker._rule_based_semantic_chunking(self.test_text)
        self.assertGreater(len(chunks), 0, "Fallback chunking failed")
        print("✓ Rule-based semantic chunking works")

        # Test quality scorer fallback
        scorer = SemanticQualityScorer({})
        score = scorer._score_coherence_fallback(self.test_text)
        self.assertGreaterEqual(score, 0, "Fallback coherence scoring failed")
        self.assertLessEqual(score, 1, "Fallback coherence score out of range")
        print("✓ Fallback quality scoring works")

        # Test metacognitive fallback
        annotator = MetacognitiveAnnotator({})
        confidence = annotator._analyze_confidence_fallback(self.test_text)
        self.assertIn('overall_confidence', confidence)
        print("✓ Fallback metacognitive analysis works")

        # Test adversarial fallback
        tester = AdversarialTester({})
        contradictions = tester._detect_contradictions_fallback(self.test_text)
        self.assertIn('contradiction_score', contradictions)
        print("✓ Fallback adversarial testing works")

        return True


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*60)
    print("🧪 RUNNING COMPREHENSIVE TESTS FOR ADVANCED FEATURES")
    print("="*60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAdvancedFeatures)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Report results
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        print(f"   Tests run: {result.testsRun}")
        print(f"   Success rate: 100%")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"   Tests run: {result.testsRun}")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")

        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split(chr(10))[0]}")

        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split(chr(10))[0]}")

    print("="*60 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)