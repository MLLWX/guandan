import random
import time
import threading

import numpy as np
import torch
import torch.multiprocessing as mp

class Buffer:
    def __init__(self, buffer_size, batch_size, device):
        _buffer = (torch.empty((buffer_size, 54+13+54), dtype=torch.float32),
                   torch.empty((buffer_size, ), dtype=torch.float32))
        if not device == "cpu":
            _buffer = [b.to(torch.device('cuda:'+str(device))).share_memory_() for b in _buffer]
        else:
            _buffer = [b.to(torch.device("cpu")).share_memory_() for b in _buffer]
        self._buffer = _buffer
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        ctx = mp.get_context("spawn")
        self._index = ctx.Value("i", 0)
        self._size = ctx.Value("i", 0)
        self.lock = ctx.RLock()
    
    def __len__(self):
        with self.lock:
            size = self._size.value
        return size
    
    @property
    def index(self):
        with self.lock:
            idx = self._index.value
        return idx

    def sample(self):
        while len(self) < self.batch_size:
            time.sleep(5)
        with self.lock:
            index = random.sample(range(self._size.value), self.batch_size)
            batch = [torch.clone(b[index]) for b in self._buffer]
        return batch
    
    def index_range(self, lens):
        index_list = np.empty((lens, ), dtype=np.int8)
        len_first = min(self.buffer_size-self._index.value, lens)
        index_list[:len_first] = range(self._index.value, self._index.value+len_first)
        if len_first < lens:
            index_list[len_first:] = range(lens - len_first)
        return index_list.tolist()
    
    def add(self, state, target):
        assert len(state) == len(target)
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        target = torch.as_tensor(target, dtype=torch.float32).to(self.device)
        with self.lock:
            if self._size.value < self.buffer_size:
                self._size.value = min(len(state)+self._size.value, self.buffer_size)
            index = self.index_range(len(state))
            self._buffer[0][index] = state
            self._buffer[1][index] = target
            self._index.value = (index[-1] + 1) % self.buffer_size
            
def act(buffer: Buffer):
    for i in range(10):
        time.sleep(2)
        data = (np.random.randn(7, 121), np.random.rand(7))
        print(f"actor: add data {data[0].shape, data[1].shape}")
        buffer.add(data[0], data[1])
        print(f"actor: buffer size {len(buffer)}, index {buffer.index}")

def learn(buffer: Buffer, idx: int=0):
    for i in range(10):
        time.sleep(2)
        batch = buffer.sample()
        print(f"learner {idx}: learn batch {batch[0].shape, batch[1].shape}")

def main():
    buffer = Buffer(64, 16, "cpu")
    ctx = mp.get_context("spawn")
    actor = ctx.Process(target=act, args=(buffer, ))
    actor.start()

    learner = threading.Thread(target=learn, args=(buffer, ))
    learner.start()

    actor.join()
    learner.join()

if __name__ == "__main__":
    main()