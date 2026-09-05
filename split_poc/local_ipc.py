"""Bounded reusable host buffers between a service and its local worker.

Cloud uses these in shm mode; opt-in pipelining also uses them on enterprise.

The caller's executor lock protects the single slot. No cross-side sharing.
Returned arrays are copied before releasing that lock, so the next request
cannot overwrite data that HTTP is still packing. Parent owns unlink.
"""
from multiprocessing import shared_memory
import numpy as np


class LocalMailbox:
    def __init__(self, names=None, capacity=128*1024**2):
        self.owner = names is None
        self.buffers = []
        try:
            for index in range(2):
                self.buffers.append(shared_memory.SharedMemory(create=self.owner, size=capacity if self.owner else 0,
                    name=None if names is None else names[index]))
        except BaseException:
            self.close()
            raise

    @property
    def names(self):
        return [buffer.name for buffer in self.buffers]

    def view(self, index, shape):
        if (len(shape)!=2 or any(type(n) is not int or n<=0 for n in shape)
                or shape[1]!=2048 or 4*shape[0]*shape[1]>self.buffers[index].size):
            raise ValueError('Invalid shared activation shape')
        return np.ndarray((2,*shape),dtype=np.float16,buffer=self.buffers[index].buf)

    def write(self,index,arrays):
        if len(arrays)!=2 or arrays[0].shape!=arrays[1].shape or any(a.dtype!=np.float16 for a in arrays):
            raise ValueError('Expected matching FP16 hidden/residual')
        shape=list(arrays[0].shape)
        target=self.view(index,shape)
        np.copyto(target[0],arrays[0])
        np.copyto(target[1],arrays[1])
        return shape

    def read(self,index,shape,copy=False):
        view=self.view(index,shape)
        return [view[i].copy() if copy else view[i] for i in range(2)]

    def close(self):
        for buffer in self.buffers:
            buffer.close()
            if self.owner:
                try:
                    buffer.unlink()
                except FileNotFoundError:
                    pass
        self.buffers=[]
