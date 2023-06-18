import os
import threading
import timeit
import numpy as np
import logging

import torch
from torch import multiprocessing as mp
from torch import nn

from .models import DecisionModel, DecisionModelSimple, TrainModelList
from .utils import get_batch, log, formatter, create_buffers, create_optimizers, act, pack_history
from ..evaluation import evaluate_for_training


def compute_loss(logits, targets):
    loss = ((logits.flatten() - targets.flatten())**2).mean()
    return loss

def learn(actor_models,
          model,
          batch,
          optimizer,
          flags,
          lock):
    """Performs a learning (optimization) step."""
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
    else:
        device = torch.device('cpu')
    actions = torch.flatten(batch['actions'].to(device), 0, 1).float()
    obs = torch.flatten(batch['obs'].to(device), 0, 1).float()
    history = pack_history(batch['history'].to(device), batch['history_lens'].to(device))

    targets = torch.flatten(batch['targets'].to(device), 0, 1)

    with lock:
        learner_outputs = model(obs, history, actions)
        loss = compute_loss(learner_outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()
        for actor_model_list in actor_models.values():
            actor_model_list.load_state_dict(model.state_dict())

def train(flags):  
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.
    """
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError("CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, f'model_{flags.train_model}.tar')))
    
    log_path = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, f'log_{flags.train_model}.txt')))
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    # Initialize actor models
    if flags.train_model == "base":
        train_model = DecisionModel
    elif flags.train_model == "base_simple":
        train_model = DecisionModelSimple
    else:
        raise NotImplementedError
    
    models = {}
    for device in device_iterator:
        model_list = TrainModelList(train_model, device=device, flags=flags)
        model_list.share_memory()
        model_list.eval()
        models[device] = model_list

    # Initialize buffers
    buffers = create_buffers(flags, device_iterator)
   
    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = {}
    full_queue = {}
        
    for device in device_iterator:
        free_queue[device] = ctx.SimpleQueue()
        full_queue[device] = ctx.SimpleQueue()

    # Learner model for training
    learner_model = train_model(device=flags.training_device)

    # Create optimizers
    optimizer = create_optimizers(flags, learner_model)

    frames = 0

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        log.info(f"load model {checkpointpath}")
        checkpoint_states = torch.load(
            checkpointpath, map_location=("cuda:"+str(flags.training_device) if flags.training_device != "cpu" else "cpu")
        )
        learner_model.load_state_dict(checkpoint_states["model_state_dict"])
        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
        for device in device_iterator:
            models[device].load_state_dict(learner_model.state_dict())
        frames = checkpoint_states["frames"]

    # Starting actor processes
    for device in device_iterator:
        num_actors = flags.num_actors
        for i in range(num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queue[device], full_queue[device], models[device], buffers[device], flags))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(device, local_lock, learn_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device], full_queue[device], buffers[device], flags, local_lock)
            learn(models, learner_model, batch, optimizer, flags, learn_lock)
            with lock:
                frames += T * B

    for device in device_iterator:
        for m in range(flags.num_buffers):
            free_queue[device].put(m)

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = threading.Lock()
    learn_lock = threading.Lock()

    for device in device_iterator:
        for _ in range(flags.num_threads):
            thread = threading.Thread(
                target=batch_and_learn, name='batch-and-learn-%d' % i, args=(device,locks[device], learn_lock))
            thread.start()
            threads.append(thread)
    
    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        torch.save({
            'model_state_dict': learner_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'flags': vars(flags),
            'frames': frames,
        }, checkpointpath)

        # Save the weights for evaluation purpose
        model_weights_dir = os.path.expandvars(os.path.expanduser(
            '%s/%s/%s' % (flags.savedir, flags.xpid, f'{flags.train_model}_weights_'+str(frames)+'.ckpt')))
        torch.save(learner_model.state_dict(), model_weights_dir)

    fps_log = []
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            start_frames = frames
            start_time = timer()
            
            win_probs, position_counts = evaluate_for_training(learner_model, flags, frames, 4, 48, if_plot=True)

            if timer() - last_checkpoint_time > flags.save_interval * 60:  
                checkpoint(frames)
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            log.info('After %i frames: @ %.1f fps (avg@ %.1f fps) \n'
                     "win probs (%s vs %s) %.2f - %.2f",
                     frames,
                     fps,
                     fps_avg,
                     flags.train_model,
                     flags.eval_model,
                     win_probs[0],
                     win_probs[1])
            position_counts_str = ""
            for i in range(4):
                position_counts_str += f"player {i}: {position_counts[i]}\n"
            log.info("position counts:\n " + position_counts_str)

    except KeyboardInterrupt:
        return 
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
