import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from retrieval import recover

class RetrievalTests(unittest.TestCase):
 def test_full_index_position_order_and_scale(self):
  e=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]])
  self.assertEqual(recover(torch.tensor([[-3.,0.],[0.,2.],[7.,0.]]),e,batch=2),[2,1,0])

if __name__=='__main__':unittest.main()
