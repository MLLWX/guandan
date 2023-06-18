import typing
import logging
import traceback

import torch 
import numpy as np

from ..env import Env, RLAgent, HISTORY_PER_PLAYER_SIZE

formatter = logging.Formatter('[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s')

log = logging.getLogger('guandanAI')
log.setLevel(logging.DEBUG)
log.propagate = False

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

log.addHandler(ch)

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]

def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch

def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
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
    T = flags.unroll_length
    obs_dim:int = 501
    buffers = {}
    for device in device_iterator:
        specs = dict(
            dones=dict(size=(T,), dtype=torch.bool),
            targets=dict(size=(T,), dtype=torch.float32),
            obs=dict(size=(T, obs_dim), dtype=torch.int8),
            history=dict(size=(T, 4, HISTORY_PER_PLAYER_SIZE, 67), dtype=torch.int8),
            history_lens=dict(size=(T, 4), dtype=torch.int8),
            actions=dict(size=(T, 67), dtype=torch.int8)
        )
        _buffers: Buffers = {key: [] for key in specs}
        for _ in range(flags.num_buffers):
            for key in _buffers:
                if not device == "cpu":
                    _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                else:
                    _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                _buffers[key].append(_buffer)
        buffers[device] = _buffers
    return buffers


def history2tensor(history):
    history_tensor = torch.stack([torch.from_numpy(_history) for _, _history in history], dim=0)
    history_lens = torch.tensor([lens for lens, _ in history], dtype=torch.int8)
    return history_tensor, history_lens

def pack_history(history_tensor, history_lens_tensor):
    history_tensor = torch.flatten(history_tensor, 0, 1).float()
    history_lens_tensor = torch.flatten(history_lens_tensor, 0, 1).cpu().to(torch.int64)
    history = [(history_tensor[:, i, ...], history_lens_tensor[:, i]) for i in range(4)]
    return history

def act(i, device, free_queue, full_queue, model_list, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    positions = [i for i in range(4)]
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        env = Env(render=False)
        agents = [RLAgent(i, model_list[i]) for i in positions]
        env.set_agents(agents)
        
        done_buf = [[] for _ in positions]
        target_buf = [[] for _ in positions]
        obs_buf = [[] for _ in positions]
        history_buf = [[] for _ in positions]
        action_buf = [[] for _ in positions]
        size_buf = [0 for _ in positions]
        with torch.no_grad():
            while True:
                env.reset()
                while not env.is_terminal():
                    trajectories = env.run(is_training=True)
                    payoffs = trajectories["payoffs"]
                    for position in positions:
                        position_states = trajectories["states"][position]
                        obs_buf[position].extend(state["obs"] for state in position_states)
                        history_buf[position].extend(state["history"] for state in position_states)

                        position_actions = trajectories["actions"][position]
                        action_buf[position].extend(position_actions)
                        
                        position_lens = len(trajectories["states"][position])
                        size_buf[position] += position_lens

                        done_buf[position].extend([False for _ in range(position_lens-1)])
                        done_buf[position].append(True)

                        target_buf[position].extend([payoffs[position] for _ in range(position_lens)])
                            
                        while size_buf[position] > T: 
                            index = free_queue.get()
                            if index is None:
                                break
                            for t in range(T):
                                buffers['dones'][index][t, ...] = done_buf[position][t]
                                buffers['targets'][index][t, ...] = target_buf[position][t]
                                buffers['actions'][index][t, ...] = torch.from_numpy(action_buf[position][t])
                                buffers['obs'][index][t, ...] = torch.from_numpy(obs_buf[position][t])
                                buffers['history'][index][t, ...],  buffers['history_lens'][index][t, ...] = history2tensor(history_buf[position][t])

                            full_queue.put(index)

                            done_buf[position] = done_buf[position][T:]
                            target_buf[position] = target_buf[position][T:]
                            action_buf[position] = action_buf[position][T:]
                            obs_buf[position] = obs_buf[position][T:]
                            history_buf[position] = history_buf[position][T:]
                            size_buf[position] -= T

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e


if __name__ == "__main__":
    pass