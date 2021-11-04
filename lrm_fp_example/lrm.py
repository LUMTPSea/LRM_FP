from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random
import numpy as np
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.nn.functional as F

from open_spiel.python import rl_agent

Transition = collections.namedtuple(
    "Transition", 
    "info_state action reward next_info_state is_final_step legal_actions_mask"
)

PolicyData = collections.namedtuple(
    "PolicyData", 
    "info_state matched_regret"
)

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9

class ReplayBuffer(object):
    def __init__(self, replay_buffer_capacity):
        self._replay_buffer_capacity = replay_buffer_capacity
        self._data = []
        self._next_entry_index = 0

    def add(self, element):
        if len(self._data) < self._replay_buffer_capacity:
            self._data.append(element)
        else:
            self._data[self._next_entry_index]  = element
            self._next_entry_index += 1
            self._next_entry_index %= self._replay_buffer_capacity
        
    def sample(self, num_samples):
        if len(self._data) < num_samples:
            raise ValueError("{} elements could not be sampled from size {}".format(num_samples, len(self._data)))
        return random.sample(self._data, num_samples)
    
    def __len__(self):
        return len(self._data)
    
    def __iter__(self):
        return iter(self._data)
    

class ReservoirBuffer(object):
    def __init__(self, reservoir_buffer_capacity):
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self._data = []
        self._add_calls = 0

    
    def add(self, element):
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            idx = random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
        self._add_calls += 1
    
    def clear(self):
        self._data = []
        self._add_calls = 0
    
    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


class SonnetLinear(nn.Module):
    def __init__(self, in_size, out_size, activate_relu=True):
        super(SonnetLinear, self).__init__()
        self._activate_relu = activate_relu
        stddev = 1.0 /math.sqrt(in_size)
        mean = 0
        lower = (-2 *stddev - mean) / stddev
        upper = (2 * stddev - mean) / stddev 

        self._weight = nn.Parameter(torch.Tensor(
            truncnorm.rvs(lower, upper, loc=mean, scale=stddev, 
            size=[out_size, in_size])))
        self._bias = nn.Parameter(torch.zeros(out_size))

    def forward(self, tensor):
        y = F.linear(tensor, self._weight, self._bias)
        return F.relu(y) if self._activate_relu else y

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activate_final=False):
        super(MLP, self).__init__()
        self._layers = []
        for size in hidden_sizes:
            self._layers.append(SonnetLinear(in_size=input_size, out_size=size))
            input_size = size
        self._layers.append(SonnetLinear(in_size=size, out_size=output_size, activate_relu=activate_final))
        self.model = nn.ModuleList(self._layers)
    
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

class LRM(rl_agent.AbstractAgent):
    def __init__(self, 
        device, 
        player_id, 
        state_representation_size, 
        num_actions, 
        hidden_layers_sizes, 
        args, 
        replay_buffer_capacity=int(1e4), 
        replay_buffer_class=ReplayBuffer, 
        ):

        self._kwargs = locals()

        self._device = device

        self._player_id = player_id
        self._num_actions = num_actions
        if isinstance(hidden_layers_sizes, int):
            hidden_layers_sizes = [hidden_layers_sizes]
        self._layers_sizes = hidden_layers_sizes
        self._batch_size = args.batch_size
        self._update_target_network_every = args.update_target_network_every
        self._learn_every = args.learn_every
        self._min_buffer_size_to_learn = args.min_buffer_size_to_learn
        self._discount_factor = args.discount_factor

        # self._epsilon_start = epsilon_start
        # self._epsilon_end = epsilon_end
        # self._epsilon_decay_duration = args.num_train_episodes

        if not isinstance(replay_buffer_capacity, int):
            raise ValueError("Replay buffer capacity not an integet.")
        
        self._replay_buffer  = replay_buffer_class(replay_buffer_capacity)
        self._policy_buffer = replay_buffer_class(replay_buffer_capacity)
        self._prev_timestep = None
        self._prev_action = None

        self._step_counter = 0

        self._last_q_loss = None
        self._last_pi_loss = None

        self._q_network = MLP(state_representation_size, self._layers_sizes, num_actions).to(self._device)

        self._target_q_network = MLP(state_representation_size, self._layers_sizes, num_actions).to(self._device)

        # self._policy_logits= MLP(state_representation_size, self._layers_sizes, num_actions).to(self._device)

        if args.loss_str == "mse":
            self.loss_class = F.mse_loss
        elif args.loss_str == "huber":
            self.loss_class = F.smooth_l1_loss
        else:
            raise ValueError("Not implemented, choose from 'mse', 'huber'.")
        
        if args.optimizer_str == "adam":
            self._q_optimzier = torch.optim.Adam(self._q_network.parameters(), lr=args.rl_q_learning_rate)
            # self._pi_optimizer = torch.optim.Adam(self._policy_logits.parameters(), lr=args.rl_pi_learning_rate)
        elif args.optimizer_str == "sgd":
            self._q_optimzier = torch.optim.SGD(self._q_network.parameters(), lr=args.rl_q_learning_rate)
            # self._pi_optimizer = torch.optim.SGD(self._policy_logits.parameters(), lr=args.rl_pi_learning_rate)
        else:
            raise ValueError("Not implemented, choose from 'adam', 'sgd'.")

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(args.seed)        



    def step(self, time_step, avg_pi, is_evaluation=False, add_transition_record=True):
        if (not time_step.last() and (time_step.is_simultaneous_move() or self._player_id == time_step.current_player())):
            info_state = time_step.observations["info_state"][self._player_id]
            legal_actions = time_step.observations["legal_actions"][self._player_id]
            # epsilon = self._get_epsilon(is_evaluation)
            action, probs = self._regret_action(info_state, legal_actions, avg_pi)
        else:
            action = None
            probs = []
        
        if not is_evaluation:
            self._step_counter += 1

            if self._step_counter % self._learn_every == 0:
                self._last_q_loss = self._update_q()
                # self._last_pi_loss = self._update_pi()
            
            if self._step_counter % self._update_target_network_every == 0:
                self._target_q_network.load_state_dict(self._q_network.state_dict())
            
            if self._prev_timestep and add_transition_record:
                self.add_transition(self._prev_timestep, self._prev_action, time_step)
            
            if time_step.last():
                self._prev_timestep = None
                self._prev_action = None
                return None
            else:
                self._prev_timestep = time_step
                self._prev_action = action
        
        return rl_agent.StepOutput(action=action, probs=probs)

    def add_transition(self, prev_timestep, prev_action, time_step):
        assert prev_timestep is not None
        legal_actions = (time_step.observations["legal_actions"][self._player_id])
        legal_actions_mask = np.zeros(self._num_actions)
        legal_actions_mask[legal_actions] = 1.0
        transition = Transition(
            info_state=prev_timestep.observations["info_state"][self._player_id][:], 
            action=prev_action, 
            reward=time_step.rewards[self._player_id], 
            next_info_state=time_step.observations["info_state"][self._player_id][:], 
            
            is_final_step=float(time_step.last()), 
            legal_actions_mask=legal_actions_mask)
        self._replay_buffer.add(transition)

    # def add_policy_data(self, info_state, matched_regret):
    #     policydata = PolicyData(
    #         info_state=info_state, 
    #         matched_regret=matched_regret)
    #     self._policy_buffer.add(policydata)
    
    def _regret_action(self, info_state, legal_actions, avg_pi):
        probs = np.zeros(self._num_actions)
        # if np.random.rand() < epsilon:
        #     action = np.random.choice(legal_actions)
        #     probs[legal_actions] = 1.0 / len(legal_actions)
        # else:
        info_state_tensor = torch.Tensor(np.reshape(info_state, [1, -1])).to(self._device)
        q_values = self._q_network(info_state_tensor).detach()[0]
        legal_q_values = q_values[legal_actions]
        # pi = F.softmax(self._policy_logits(info_state_tensor), dim=1).detach()[0]
        legal_pi = avg_pi[legal_actions]
        baseline = torch.sum(torch.mul(legal_q_values, legal_pi))
        
        regret = np.zeros(self._num_actions)
        for action in legal_actions:
            regret[action] = F.relu(q_values[action] - baseline)
        cumulative_regret = np.sum(regret)
        matched_regret = np.array([0.] * self._num_actions)
        for action in legal_actions:
            if cumulative_regret > 0:
                matched_regret[action] = regret[action] / cumulative_regret
            else:
                matched_regret[action] = 1.0 / len(legal_actions)
        # self.add_policy_data(info_state, matched_regret)
        action = np.random.choice(self._num_actions, p=matched_regret)
        probs[action] = 1.0

        return action, probs


    
    def _update_q(self):
        if (len(self._replay_buffer) < self._batch_size or 
            len(self._replay_buffer) <  self._min_buffer_size_to_learn):
            return None
        
        transitions = self._replay_buffer.sample(self._batch_size)
        info_states = torch.Tensor([t.info_state for t in transitions]).to(self._device)
        actions = torch.LongTensor([t.action for t in transitions])
        rewards = torch.Tensor([t.reward for t in transitions]).to(self._device)
        next_info_states = torch.Tensor([t.next_info_state for t in transitions]).to(self._device)
        are_final_steps = torch.Tensor([t.is_final_step for t in transitions]).to(self._device)
        legal_actions_mask = torch.LongTensor([t.legal_actions_mask for t in transitions]).to(self._device)

        self._q_values = self._q_network(info_states)
        self._target_q_values = self._target_q_network(next_info_states).detach()
        # self._next_pi = F.softmax(self._policy_logits(next_info_states).detach(), dim=1)

        self._legal_target_q = torch.mul(self._target_q_values, legal_actions_mask)
        # self._legal_next_pi = torch.mul(self._next_pi, legal_actions_mask)

        # V_baseline = torch.sum(torch.mul(self._legal_target_q, self._legal_next_pi), dim=1)
        illegal_actions = 1 - legal_actions_mask
        illegal_logits = illegal_actions * ILLEGAL_ACTION_LOGITS_PENALTY
        max_next_q = torch.max(self._target_q_values + illegal_logits, dim=1)[0]
        target = (rewards  + (1 - are_final_steps) * self._discount_factor * max_next_q)
        # target = rewards + (1 - are_final_steps) * self._discount_factor * V_baseline
        
        action_indices = torch.stack([torch.arange(self._q_values.shape[0]), actions], dim=0)
        predictions = self._q_values[list(action_indices)]
        q_loss = self.loss_class(predictions, target)

        self._q_optimzier.zero_grad()
        q_loss.backward()
        self._q_optimzier.step()

        return q_loss

    
    @property
    def q_values(self):
        return self._q_values
    
    @property
    def replay_buffer(self):
        return self._replay_buffer
    
    @property
    def policy_buffer(self):
        return self._policy_buffer

    @property
    def loss(self):
        return self._last_q_loss, self._last_pi_loss

    @property
    def prev_timestep(self):
        return self._prev_timestep

    @property
    def prev_action(self):
        return self._prev_action

    @property
    def step_counter(self):
        return self._step_counter

    def get_weights(self):
        variables = []
        q_variables = [m.weight for m in self._q_network.model]
        pi_varaibles = [m.weight for m in self._policy_logits.model]
        target_q_variables = [m.weight for m in self._target_q_network.model]
        variables.append(q_variables, pi_varaibles, target_q_variables)
    
    def copy_with_noise(self, sigma=0.0, copy_weights=True):
        _ = self._kwargs.pop("self", None)
        copied_object = RegretDQN(**self._kwargs)
        q_network = getattr(copied_object, "_q_network")
        target_q_network = getattr(copied_object, "_target_q_network")
        pi_network = getattr(copied_object, "_pi_logits")
        
        if copy_weights:
            with torch.no_grad():
                for q_model in q_network.model:
                    q_model.weight *= (1 + sigma * torch.randn(q_model.weight.shape))
                for tq_model in target_q_network.model:
                    tq_model.weight *= (1 + sigma * torch.randn(tq_model.weigth.shape))
                for pi_model in pi_network.model:
                    pi_model.weight *= (1 + sigma * torch.randn(pi_model.weight.shape))
        return copied_object
























    


        
