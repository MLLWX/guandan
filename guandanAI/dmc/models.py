import os

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DecisionModel(nn.Module):
    def __init__(self, device="cpu", flags=None):
        super().__init__()
        self.lstm = nn.LSTM(67, 32, batch_first=True)
        self.dense1 = nn.Linear(501 + 32*4 + 67, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 128)
        self.dense4 = nn.Linear(128, 64)
        self.dense5 = nn.Linear(64, 32)
        self.dense6 = nn.Linear(32, 1)

        if not device == "cpu":
            if not device.startswith("cuda"):
                device = 'cuda:' + str(device)
        self.to(device)
        self.device = device
        self.flags = flags

    def forward(self, obs, history, actions):
        lstm_out = []
        for _history, _len in history:
            pack_history = pack_padded_sequence(_history, _len, enforce_sorted=False, batch_first=True)
            _lstm_out, (h_n, _) = self.lstm(pack_history)
            _lstm_out, out_len = pad_packed_sequence(_lstm_out, batch_first=True)
            _lstm_out = _lstm_out[range(len(_history)),out_len-1,:]
            lstm_out.append(_lstm_out)
        
        x = torch.cat([obs, *lstm_out, actions], dim=-1)
        x = x.to(torch.float32)

        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        return x
    
    def step(self, obs, legal_actions, history, is_train):
        ''' Predict the action given the current state for evaluation.

        Args:
            obs (ndarray): An ndarray that represents the current observation
            legal_actions (ndarray): legal actions to choose
            history (list of tuple): each player's history action and history length
            is_train (bool): in train step or evaluation step
        Returns:
            action_index (int): The action index in legal_actions predicted model
        '''
        obs_input = np.repeat(obs[np.newaxis, :], len(legal_actions), axis=0)
        history_input = [(torch.as_tensor(np.repeat(_history[np.newaxis, :, :], len(legal_actions), axis=0), 
                                          dtype=torch.float32, device=self.device), [_len for _ in range(len(legal_actions))]) 
                         for _len, _history in history]
        # to tensor
        legal_actions = torch.as_tensor(legal_actions, dtype=torch.float32, device=self.device)
        obs_input = torch.as_tensor(obs_input, dtype=torch.float32, device=self.device)

        x = self(obs_input, history_input, legal_actions)

        if is_train and self.flags is not None and self.flags.exp_epsilon > 0 and np.random.rand() < self.flags.exp_epsilon:
            action_index = torch.randint(x.shape[0], (1,)).item()
        else:
            action_index = torch.argmax(x,dim=0).item()
        return action_index

class DecisionModelTribute(DecisionModel):
    def __init__(self, device="cpu", flags=None):
        super().__init__(device, flags)
        self.tribute_model = TributeModel(device=self.device)

    def load_base_model(self, base_model_path):
        if os.path.exists(base_model_path):
            print("load base model")
            base_state_dict = torch.load(base_model_path, map_location=self.device)
            self_state_dict = self.state_dict()
            self_state_dict.update(base_state_dict)
            self.load_state_dict(self_state_dict)
    def load_tribute_state_dict(self, tribute_state_dict):
        self.tribute_model.load_state_dict(tribute_state_dict)
    
class DecisionModelSimple(nn.Module):
    # Model without LSTM
    def __init__(self, device="cpu", flags=None):
        super().__init__()
        self.dense1 = nn.Linear(501 + 67, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 128)
        self.dense4 = nn.Linear(128, 64)
        self.dense5 = nn.Linear(64, 32)
        self.dense6 = nn.Linear(32, 1)

        if not device == "cpu":
            if not device.startswith("cuda"):
                device = 'cuda:' + str(device)
        self.to(device)
        self.device = device
        self.flags = flags

    def forward(self, obs, history, actions):
        x = torch.cat([obs, actions], dim=-1)
        x = x.to(torch.float32)

        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        return x
    
    def step(self, obs, legal_actions, history, is_train):
        ''' Predict the action given the current state for evaluation.

        Args:
            obs (ndarray): An ndarray that represents the current observation
            legal_actions (ndarray): legal actions to choose
            history (list of tuple): each player's history action and history length
            is_train (bool): in train step or evaluation step
        Returns:
            action_index (int): The action index in legal_actions predicted model
        '''
        obs_input = np.repeat(obs[np.newaxis, :], len(legal_actions), axis=0)
        # to tensor
        legal_actions = torch.as_tensor(legal_actions, dtype=torch.float32, device=self.device)
        obs_input = torch.as_tensor(obs_input, dtype=torch.float32, device=self.device)

        x = self(obs_input, None, legal_actions)

        if is_train and self.flags is not None and self.flags.exp_epsilon > 0 and np.random.rand() < self.flags.exp_epsilon:
            action_index = torch.randint(x.shape[0], (1,)).item()
        else:
            action_index = torch.argmax(x,dim=0).item()
        return action_index
    
class TributeModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.dense1 = nn.Linear(54+13+54, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)
        self.dense4 = nn.Linear(32, 1)

        if not device == "cpu":
            if not device.startswith("cuda"):
                device = 'cuda:' + str(device)
        self.to(device)
        self.device = device

    def forward(self, X):
        X = self.dense1(X)
        X = torch.relu(X)
        X = self.dense2(X)
        X = torch.relu(X)
        X = self.dense3(X)
        X = torch.relu(X)
        X = self.dense4(X)
        return X
    
    def pay_tribute(self, hand_cards_array, officer_array, actions):
        hand_cards_input = np.repeat(hand_cards_array[np.newaxis, :], len(actions), axis=0)
        hand_cards_input = torch.as_tensor(hand_cards_input, dtype=torch.float32, device=self.device)
        officer_input = np.repeat(officer_array[np.newaxis, :], len(actions), axis=0)
        officer_input = torch.as_tensor(officer_input, dtype=torch.float32, device=self.device)
        actions_input = torch.zeros((len(actions), 54), dtype=torch.float32, device=self.device)
        actions_input[range(len(actions)), actions] = 1
        rst = self(torch.cat((hand_cards_input, officer_input, actions_input), dim=-1))
        return torch.argmax(rst, dim=0).item()

    

class TrainModelList:
    """
    The wrapper for the four models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, model, device=0, flags=None):
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self._models = [model(device, flags).to(torch.device(device)) for _ in range(4)]

    def load_state_dict(self, state_dict):
        for model in self._models:
            model.load_state_dict(state_dict=state_dict)
    
    def load_base_model(self, base_model_path):
        if isinstance(self._models[0], DecisionModelTribute):
            for model in self._models:
                model.load_base_model(base_model_path)
    
    def load_tribute_state_dict(self, tribute_state_dict):
        if isinstance(self._models[0], DecisionModelTribute):
            for model in self._models:
                model.load_tribute_state_dict(tribute_state_dict)

    def share_memory(self):
        for model in self._models:
            model.share_memory()

    def eval(self):
        for model in self._models:
            model.eval()

    def __len__(self):
        return len(self._models)
    
    def __iter__(self):
        return iter(self._models)
    
    def __getitem__(self, idx):
        return self._models[idx]

    def __setitem__(self, idx, model):
        self._models[idx] = model

    
