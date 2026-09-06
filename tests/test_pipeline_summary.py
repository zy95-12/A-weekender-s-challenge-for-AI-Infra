import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_pipeline import union,intersections,measure

class IntervalTests(unittest.TestCase):
    def test_overlapping_ranks_and_transfers_are_not_double_counted(self):
        self.assertEqual(union([(0,10),(2,8),(8,20),(30,40)]),[(0,20),(30,40)])
        self.assertEqual(intersections([(0,20),(30,40)],[(5,15),(10,35)]),[(5,20),(30,35)])
        self.assertEqual(measure([(0,10_000_000),(5_000_000,15_000_000)]),15)
