import random
import time
import logging
import traceback

import torch 
import torch.multiprocessing as mp
import numpy as np

from ..env import Env, RLAgent

formatter = logging.Formatter('[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s')

log = logging.getLogger('guandanAI')
log.setLevel(logging.DEBUG)
log.propagate = False

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

log.addHandler(ch)

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

def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    optimizer = torch.optim.RMSprop(
        learner_model.tribute_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)
    return optimizer

def create_buffers(flags, device_iterator):
    """
    We create buffers for different devices (i.e., GPU). That is, each device
    will have buffer.
    """
    batch_size = flags.batch_size
    buffer_size = flags.buffer_size
    buffers = {}
    for device in device_iterator:        
        buffers[device] = Buffer(buffer_size, batch_size, "cpu")
    return buffers

def act(i, device, model_list, buffer):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    positions = [i for i in range(4)]
    try:
        log.info('Device %s Actor %i started.', str(device), i)

        env = Env(render=False)
        agents = [RLAgent(i, model_list[i]) for i in positions]
        env.set_agents(agents)

        with torch.no_grad():
            while True:
                env.reset()
                state, target = [], []
                while not env.is_terminal():
                    trajectories = env.run(is_training=True)
                    payoffs = trajectories["payoffs"]
                    for i, agent in enumerate(agents):
                        if agent.tribute_input is not None:
                            state.append(agent.tribute_input)
                            target.append(payoffs[i])
                if state:
                    buffer.add(np.array(state), np.array(target))

                

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e


if __name__ == "__main__":
    pass