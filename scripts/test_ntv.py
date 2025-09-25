#!/usr/bin/env python
"""
Test extraction from n-tv.de (German news website)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core import DatasetCreator
from src.extractors import WebExtractor
import time
import json
from datetime import datetime


def test_ntv_extraction():
    """Test extraction from www.n-tv.de German news website"""

    print("\n" + "="*60)
    print("Testing n-tv.de Extraction")
    print("="*60)

    # Test URLs from n-tv.de
    test_urls = [
        ("Main Page", "https://www.n-tv.de"),
        ("Politics Section", "https://www.n-tv.de/politik/"),
        ("Economy Section", "https://www.n-tv.de/wirtschaft/"),
        ("Technology Section", "https://www.n-tv.de/technik/"),
    ]

    results = []

    for name, url in test_urls:
        print(f"\n[Testing] {name}: {url}")
        print("-" * 40)

        try:
            # Create extractor with appropriate settings
            extractor = WebExtractor({
                'max_depth': 0,  # Just the main page
                'max_pages': 1,
                'rate_limit': 0.5,  # Be respectful
                'timeout': 30,
                'javascript_rendering': False,  # Try static first
                'user_agent': 'Dataseter/1.0 Educational Testing'
            })

            start_time = time.time()
            result = extractor.extract(url)
            elapsed = time.time() - start_time

            if result.get('text'):
                text_length = len(result['text'])

                # Extract some metadata
                metadata = result.get('metadata', {})

                print(f"SUCCESS: Extracted {text_length:,} characters in {elapsed:.2f}s")
                print(f"  Title: {metadata.get('title', 'N/A')[:80]}")
                print(f"  Language: {metadata.get('language', 'N/A')}")

                # Show sample of content
                sample = result['text'][:500].replace('\n', ' ')
                print(f"  Sample: {sample[:200]}...")

                # Count German-specific elements
                german_chars = sum(1 for c in result['text'] if c in 'äöüÄÖÜß')
                print(f"  German characters found: {german_chars}")

                results.append({
                    'name': name,
                    'url': url,
                    'status': 'success',
                    'chars': text_length,
                    'time': elapsed,
                    'title': metadata.get('title', ''),
                    'language': metadata.get('language', ''),
                    'german_chars': german_chars
                })
            else:
                error = result.get('error', 'No content extracted')
                print(f"FAILED: {error}")

                results.append({
                    'name': name,
                    'url': url,
                    'status': 'failed',
                    'error': error,
                    'time': elapsed
                })

        except Exception as e:
            print(f"ERROR: {str(e)}")
            results.append({
                'name': name,
                'url': url,
                'status': 'error',
                'error': str(e)
            })

    return results


def test_ntv_with_dataset_creator():
    """Test n-tv.de with full DatasetCreator pipeline"""

    print("\n" + "="*60)
    print("Testing n-tv.de with DatasetCreator Pipeline")
    print("="*60)

    creator = DatasetCreator()

    # Add n-tv.de website with depth crawling
    print("\nAdding n-tv.de to dataset creator...")
    creator.add_website("https://www.n-tv.de", max_depth=1)

    print("Processing with quality filters and chunking...")

    try:
        start_time = time.time()

        dataset = creator.process(
            chunk_size=500,
            overlap=50,
            remove_pii=True,
            quality_threshold=0.6,
            remove_duplicates=True,
            parallel=False  # Single thread for testing
        )

        elapsed = time.time() - start_time

        print(f"\n[RESULTS]")
        print(f"  Documents extracted: {len(dataset)}")
        print(f"  Processing time: {elapsed:.2f}s")

        if dataset.statistics:
            stats = dataset.statistics
            print(f"  Total text length: {stats.get('total_text_length', 0):,} chars")
            print(f"  Total words: {stats.get('total_words', 0):,}")
            print(f"  Vocabulary size: {stats.get('vocabulary_size', 0):,}")

            if 'quality_stats' in stats and stats['quality_stats']:
                print(f"  Average quality: {stats['quality_stats'].get('mean', 0):.2f}")

            if 'languages' in stats and stats['languages']:
                print(f"  Languages detected: {dict(list(stats['languages'].items())[:3])}")

        # Check for German content
        if len(dataset) > 0:
            first_doc = dataset.documents[0]
            german_chars = sum(1 for c in first_doc.text if c in 'äöüÄÖÜß')
            print(f"  German characters in first doc: {german_chars}")

            # Show sample
            print(f"\n[SAMPLE CONTENT]")
            sample = first_doc.text[:300].replace('\n', ' ')
            print(f"  {sample}...")

            # Save sample output
            output_file = "ntv_dataset_sample.jsonl"
            dataset.to_jsonl(output_file)
            print(f"\n[OUTPUT] Sample dataset saved to {output_file}")

        return dataset

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        return None


def test_ntv_deep_crawl():
    """Test deep crawling of n-tv.de (multiple pages)"""

    print("\n" + "="*60)
    print("Testing n-tv.de Deep Crawl (Multiple Pages)")
    print("="*60)

    creator = DatasetCreator()

    # Add with deeper crawling but limited pages
    creator.add_website("https://www.n-tv.de/politik/", max_depth=2)

    print("\nCrawling n-tv.de/politik with depth=2...")
    print("This may take a while...")

    try:
        dataset = creator.process(
            chunk_size=1000,
            overlap=100,
            quality_threshold=0.7,
            parallel=True,
            max_workers=2
        )

        print(f"\n[DEEP CRAWL RESULTS]")
        print(f"  Pages extracted: {len(dataset)}")

        if dataset.statistics and 'sources' in dataset.statistics:
            sources = dataset.statistics['sources']
            print(f"  Unique URLs crawled: {len(sources)}")
            print("\n  Sample URLs:")
            for url in list(sources.keys())[:5]:
                print(f"    - {url[:80]}...")

        return dataset

    except Exception as e:
        print(f"\nERROR in deep crawl: {str(e)}")
        return None


def main():
    """Main test execution"""

    print("""
    ===============================================
    n-tv.de Website Extraction Test Suite
    ===============================================

    Testing German news website extraction
    Website: www.n-tv.de
    """)

    # Run tests
    test_results = {}

    # Test 1: Basic extraction
    print("\n[TEST 1] Basic Page Extraction")
    basic_results = test_ntv_extraction()
    test_results['basic'] = basic_results

    # Test 2: Full pipeline
    print("\n[TEST 2] Full Pipeline Processing")
    dataset = test_ntv_with_dataset_creator()
    test_results['pipeline'] = {
        'success': dataset is not None,
        'documents': len(dataset) if dataset else 0
    }

    # Test 3: Deep crawl (optional - takes longer)
    user_input = input("\n[TEST 3] Run deep crawl test? (y/n): ")
    if user_input.lower() == 'y':
        deep_dataset = test_ntv_deep_crawl()
        test_results['deep_crawl'] = {
            'success': deep_dataset is not None,
            'documents': len(deep_dataset) if deep_dataset else 0
        }

    # Save test results
    results_file = "ntv_test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)

    # Summary
    if test_results.get('basic'):
        success_count = sum(1 for r in test_results['basic'] if r.get('status') == 'success')
        print(f"Basic extraction: {success_count}/{len(test_results['basic'])} successful")

    if test_results.get('pipeline'):
        if test_results['pipeline']['success']:
            print(f"Pipeline processing: SUCCESS ({test_results['pipeline']['documents']} documents)")
        else:
            print("Pipeline processing: FAILED")

    if test_results.get('deep_crawl'):
        if test_results['deep_crawl']['success']:
            print(f"Deep crawl: SUCCESS ({test_results['deep_crawl']['documents']} documents)")
        else:
            print("Deep crawl: FAILED")

    print(f"\nResults saved to: {results_file}")
    print("\nTest complete!")


if __name__ == "__main__":
    main()