import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from metrics import score

class MetricsTests(unittest.TestCase):
 def test_position_alignment_not_bag_of_tokens(self):
  r=score([1,2,3],[2,3,1]);self.assertEqual(r['token_accuracy'],0);self.assertFalse(r['exact_sequence_match'])
 def test_frequency_weighted_and_type_macro_are_distinct(self):
  r=score([1,1,1,2],[1,1,1,3],seen_token_ids={2})
  self.assertEqual(r['token_accuracy'],.75);self.assertEqual(r['macro_token_type_accuracy'],.5)
  self.assertEqual(r['training_seen_token_accuracy'],0)
 def test_exact_sequence_requires_every_position(self):
  r=score([1,2],[1,2]);self.assertTrue(r['exact_sequence_match']);self.assertEqual(r['token_accuracy'],1)
 def test_mismatched_lengths_rejected(self):
  with self.assertRaises(AssertionError):score([1],[1,2])

if __name__=='__main__':unittest.main()
