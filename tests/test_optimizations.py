import unittest
import numpy as np
from multiprocessing import shared_memory
from split_poc.local_ipc import LocalMailbox
from split_poc.wire import pack, unpack


class OptimizationTests(unittest.TestCase):
    def test_fast_wire_identical_bytes(self):
        for n in (1,4,512,2048):
            arrays=[np.arange(n*2048,dtype=np.float16).reshape(n,2048),np.full((n,2048),-.5,np.float16)]
            self.assertEqual(pack({'a':1},arrays),pack({'a':1},arrays,fast=True))

    def test_fast_wire_noncontiguous(self):
        arrays=[np.zeros((4,4096),np.float16)[:,::2]]*2
        self.assertEqual(pack({},arrays),pack({},arrays,fast=True))

    def test_mailbox_ownership_and_response_lifetime(self):
        owner=LocalMailbox(capacity=32768)
        names=owner.names
        peer=LocalMailbox(names)
        try:
            arrays=[np.full((4,2048),value,np.float16) for value in (.25,-.5)]
            shape=owner.write(0,arrays)
            returned=peer.read(0,shape)
            np.testing.assert_array_equal(returned[1],arrays[1])
            peer.write(1,returned)
            saved=owner.read(1,shape,copy=True)
            peer.write(1,[np.zeros_like(a) for a in arrays])
            np.testing.assert_array_equal(saved[1],arrays[1])
            del returned
            with self.assertRaises(ValueError):
                peer.read(0,[5,2048])
        finally:
            peer.close()
            owner.close()
        for name in names:
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)


if __name__=='__main__':
    unittest.main()
