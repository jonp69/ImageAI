import unittest
from collections import defaultdict
from extract_largeamount_of_metadata import UnparsedMetadataScanner

class TestUnparsedMetadataScannerInit(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.scanner = UnparsedMetadataScanner()
    
    def test_stats_initialization(self):
        """Test that stats dictionary is properly initialized with correct structure."""
        expected_stats_keys = [
            'total_files', 'scanned_files', 'skipped_small_files', 
            'files_with_metadata', 'unparsed_files', 'ai_generated',
            'errors', 'source_breakdown', 'unparsed_chunks', 
            'large_chunks', 'suspicious_files', 'parsing_times'
        ]
        
        # Check all expected keys exist
        for key in expected_stats_keys:
            self.assertIn(key, self.scanner.stats)
        
        # Check initial values
        self.assertEqual(self.scanner.stats['total_files'], 0)
        self.assertEqual(self.scanner.stats['scanned_files'], 0)
        self.assertEqual(self.scanner.stats['skipped_small_files'], 0)
        self.assertEqual(self.scanner.stats['files_with_metadata'], 0)
        self.assertEqual(self.scanner.stats['unparsed_files'], 0)
        self.assertEqual(self.scanner.stats['ai_generated'], 0)
        self.assertEqual(self.scanner.stats['errors'], 0)
        
        # Check collection types
        self.assertIsInstance(self.scanner.stats['source_breakdown'], defaultdict)
        self.assertIsInstance(self.scanner.stats['unparsed_chunks'], defaultdict)
        self.assertIsInstance(self.scanner.stats['large_chunks'], list)
        self.assertIsInstance(self.scanner.stats['suspicious_files'], list)
        self.assertIsInstance(self.scanner.stats['parsing_times'], list)
        
        # Check that lists are empty initially
        self.assertEqual(len(self.scanner.stats['large_chunks']), 0)
        self.assertEqual(len(self.scanner.stats['suspicious_files']), 0)
        self.assertEqual(len(self.scanner.stats['parsing_times']), 0)
    
    def test_known_ai_patterns_initialization(self):
        """Test that known AI patterns dictionary is properly initialized."""
        expected_platforms = [
            'comfyui', 'automatic1111', 'tensorart', 'midjourney', 
            'dalle', 'stablediffusion', 'novelai', 'invokeai', 
            'fooocus', 'forge', 'vlad'
        ]
        
        # Check all expected platforms exist
        for platform in expected_platforms:
            self.assertIn(platform, self.scanner.known_ai_patterns)
            self.assertIsInstance(self.scanner.known_ai_patterns[platform], list)
            self.assertGreater(len(self.scanner.known_ai_patterns[platform]), 0)
        
        # Test specific patterns
        self.assertIn('workflow', self.scanner.known_ai_patterns['comfyui'])
        self.assertIn('parameters', self.scanner.known_ai_patterns['automatic1111'])
        self.assertIn('midjourney', self.scanner.known_ai_patterns['midjourney'])
    
    def test_suspicious_patterns_initialization(self):
        """Test that suspicious patterns list is properly initialized."""
        expected_patterns = [
            'steps', 'sampler', 'cfg', 'seed', 'model', 'lora', 
            'embedding', 'checkpoint', 'vae', 'scheduler', 
            'guidance', 'denoise', 'clip_skip', 'eta', 
            'negative', 'prompt', 'width', 'height'
        ]
        
        self.assertIsInstance(self.scanner.suspicious_patterns, list)
        
        # Check all expected patterns exist
        for pattern in expected_patterns:
            self.assertIn(pattern, self.scanner.suspicious_patterns)
        
        # Check that all patterns are strings
        for pattern in self.scanner.suspicious_patterns:
            self.assertIsInstance(pattern, str)
            self.assertGreater(len(pattern), 0)
    
    def test_multiple_instances_independence(self):
        """Test that multiple scanner instances don't share mutable state."""
        scanner1 = UnparsedMetadataScanner()
        scanner2 = UnparsedMetadataScanner()
        
        # Modify one instance
        scanner1.stats['total_files'] = 100
        scanner1.stats['large_chunks'].append('test_chunk')
        scanner1.suspicious_patterns.append('test_pattern')
        
        # Check that other instance is unaffected
        self.assertEqual(scanner2.stats['total_files'], 0)
        self.assertEqual(len(scanner2.stats['large_chunks']), 0)
        self.assertNotIn('test_pattern', scanner2.suspicious_patterns)
    
    def test_defaultdict_behavior(self):
        """Test that defaultdict fields behave correctly."""
        # Test source_breakdown defaultdict
        self.assertEqual(self.scanner.stats['source_breakdown']['new_key'], 0)
        self.assertIsInstance(self.scanner.stats['source_breakdown']['new_key'], int)
        
        # Test unparsed_chunks defaultdict  
        self.assertEqual(len(self.scanner.stats['unparsed_chunks']['new_chunk']), 0)
        self.assertIsInstance(self.scanner.stats['unparsed_chunks']['new_chunk'], list)


if __name__ == '__main__':
    unittest.main()