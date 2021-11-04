from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random
from time import time
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

    def clear(self):
        self._data = []
    
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

class ARM(rl_agent.AbstractAgent):
    def __init__(self, 
        device, 
        player_id, 
        state_representation_size, 
        num_actions, 
        hidden_layers_sizes, 
        args, 
        transition_buffer_class=ReplayBuffer
        ):

        self._kwargs = locals()

        self._device = device
        self._player_id = player_id
        self._num_actions = num_actions
        if isinstance(hidden_layers_sizes, int):
            hidden_layers_sizes = [hidden_layers_sizes]
        self._layers_sizes = hidden_layers_sizes
        self._replay_buffer_capacity = args.replay_buffer_capacity

        self._transition_data = transition_buffer_class(self._replay_buffer_capacity)
        self._train_batch_size = args.train_batch_size
        self._learn_every = args.learn_every

        self._prev_timestep = None
        self._prev_action = None

        self._step_counter = 0
        self._num_learn_steps = 0
        self._last_loss_value = None
        self._arm_target_step_size = args.arm_target_step_size
        self._batch_size = args.batch_size

        self._q_loss = None
        self._v_loss = None

        self._min_buffer_size_to_learn = args.min_buffer_size_to_learn
        self._extra_discount = args.discount_factor
        self._critic_update_num = args.critic_update_num
        self._q_network = MLP(state_representation_size, self._layers_sizes, num_actions).to(self._device)
        self._prev_q_network = MLP(state_representation_size, self._layers_sizes, num_actions).to(self._device)

        self._v_network = MLP(state_representation_size, self._layers_sizes, 1).to(self._device)
        self._prev_v_network = MLP(state_representation_size, self._layers_sizes, 1).to(self._device)

        self._tg_v_network = MLP(state_representation_size, self._layers_sizes, 1).to(self._device)
        self._tg_v_network.load_state_dict(self._v_network.state_dict())

        if args.loss_str == "mse":
            self.loss_class = F.mse_loss
        elif args.loss_str == "huber":
            self.loss_class = F.smooth_l1_loss
        else:
            raise ValueError("Not implemented, choose from 'mse', 'huber'.")
        
        if args.optimizer_str == "adam":
            self._q_optimzier = torch.optim.Adam(self._q_network.parameters(), lr=args.rl_q_learning_rate)
            self._v_optimizer = torch.optim.Adam(self._v_network.parameters(), lr=args.rl_v_learning_rate)
        elif args.optimizer_str == "sgd":
            self._q_optimzier = torch.optim.SGD(self._q_network.parameters(), lr=args.rl_q_learning_rate)
            self._v_optimizer = torch.optim.SGD(self._v_network.parameters(), lr=args.rl_v_learning_rate)
        else:
            raise ValueError("Not implemented, choose from 'adam', 'sgd'.")


    def step(self, time_step, is_evaluation=False):
        if (not time_step.last() and (time_step.is_simultaneous_move() or self._player_id == time_step.current_player())):
            info_state = time_step.observations["info_state"][self._player_id]
            legal_actions = time_step.observations["legal_actions"][self._player_id]
            action, probs = self._arm_action(info_state, legal_actions)
        else:
            action = None
            probs = []
        
        if not is_evaluation:
            self._step_counter += 1

            # if self._step_counter % self._learn_every == 0:
            self._q_loss, self._v_loss = self._critic_update()
            
            if self._prev_timestep:
                self.add_transition(self._prev_timestep, self._prev_action, time_step)
            
            if time_step.last():
                self._prev_timestep = None
                self._prev_action = None
                return 
            else:
                self._prev_timestep = time_step
                self._prev_action = action
        return rl_agent.StepOutput(action=action, probs=probs)

# -------------------------------------------------------------------------------------------------------------------------------------        

    # def step(self, time_step, is_evaluation=False):
    #     if (not time_step.last() and (time_step.is_simultaneous_move() or self._player_id == time_step.current_player())):
    #         info_state = time_step.observations["info_state"][self._player_id]
    #         legal_actions = time_step.observations["legal_actions"][self._player_id]
    #         # epsilon = self._get_epsilon(is_evaluation)
    #         action, probs = self._arm_action(info_state, legal_actions)
    #     else:
    #         action = None
    #         probs = []
        
    #     if not is_evaluation:
    #         self._step_counter += 1
            
    #         if self._prev_timestep:
    #             self.add_transition(self._prev_timestep, self._prev_action, time_step)
                        
    #         if time_step.last():   
    #             if len(self._transition_data) >= self._buffer_size_to_learn: # buffer_size_to_learn就代表了这个buffer的容量了
    #                 self._last_q_loss, self._last_v_loss = self._critic_update() # 在这个函数里执行更新v网络，更新q网络和更新tgv网络的步骤
    #                 self._num_learn_steps += 1
    #                 self._transition_data.clear()           
    #             self._prev_timestep = None
    #             self._prev_action = None
    #             return 
    #         else:
    #             self._prev_timestep = time_step
    #             self._prev_action = action

    #         # if self._step_counter % self._update_target_network_every == 0:
    #         #     self._prev_q_network.load_state_dict(self._q_network.state_dict())
    #         #     self._prev_v_network.load_state_dict(self._v_network.state_dict())
            

    #     return rl_agent.StepOutput(action=action, probs=probs)
                    
# --------------------------------------------------------------------------------------------------------------------
        #     if self._step_counter % self._learn_every == 0:
        #         self._last_q_loss = self._update_q()
        #         self._last_pi_loss = self._update_pi() 
            
        #     if self._step_counter % self._update_target_network_every == 0:
        #         self._target_q_network.load_state_dict(self._q_network.state_dict())
            
        #     if self._prev_timestep and add_transition_record:
        #         self.add_transition(self._prev_timestep, self._prev_action, time_step)
            
        #     if time_step.last():
        #         self._prev_timestep = None
        #         self._prev_action = None
        #         return None
        #     else:
        #         self._prev_timestep = time_step
        #         self._prev_action = action
        
        # return rl_agent.StepOutput(action=action, probs=probs)

    def add_transition(self, prev_time_step, prev_action, time_step):
        assert prev_time_step is not None
        legal_actions = (time_step.observations["legal_actions"][self._player_id])
        legal_actions_mask = np.zeros(self._num_actions)
        legal_actions_mask[legal_actions] = 1.0
        transition = Transition(
            info_state=prev_time_step.observations["info_state"][self._player_id][:], 
            action=prev_action, 
            reward=time_step.rewards[self._player_id], 
            next_info_state=time_step.observations["info_state"][self._player_id][:], 
            is_final_step=float(time_step.last()), 
            legal_actions_mask=legal_actions_mask
            )
        self._transition_data.add(transition)

    def _critic_update(self):
        if (len(self._transition_data) < self._batch_size or
            len(self._transition_data) < self._min_buffer_size_to_learn):
            return None, None

        transitions = self._transition_data.sample(self._batch_size)
        info_states = torch.Tensor([t.info_state for t in transitions]).to(self._device)
        actions = torch.LongTensor([t.action for t in transitions]).to(self._device)
        rewards = torch.Tensor([t.reward for t in transitions]).to(self._device)
        next_info_states = torch.Tensor([t.next_info_state for t in transitions]).to(self._device)
        are_final_steps = torch.Tensor([t.is_final_step for t in transitions]).to(self._device)
        legal_actions_mask = torch.LongTensor([t.legal_actions_mask for t in transitions]).to(self._device)
        # illegl_actions = 1 - legal_actions_mask
        # illegl_logits = illegl_actions * ILLEGAL_ACTION_LOGITS_PENALTY

        tgv = self._tg_v_network(next_info_states).detach() 
        tgv_next = (1.0 - are_final_steps) * torch.squeeze(tgv, 1)

        target_v = rewards + self._extra_discount * tgv_next

        # prev_q = self._prev_q_network(info_states).detach()
        # prev_q_action = torch.gather(prev_q, 1, torch.unsqueeze(actions, 1))
        # prev_v = self._prev_v_network(info_states).detach()
        q_value = self._q_network(info_states).detach()
        q_action_value = torch.gather(q_value, 1, torch.unsqueeze(actions, 1))
        v_value = self._v_network(info_states).detach()

        if self._num_learn_steps == 0:
            target_q = rewards + self._extra_discount * tgv_next
        else:
            target_q = torch.clamp(torch.squeeze(q_action_value - v_value, 1), min=0.0) + rewards + self._extra_discount * tgv_next

        v_predictions = torch.squeeze(self._v_network(info_states), 1)
        q_predictions = torch.squeeze(torch.gather(self._q_network(info_states), 1, torch.unsqueeze(actions, 1)), 1)
        
        target_q = target_q.detach()
        target_v = target_v.detach()
        v_loss = self.loss_class(v_predictions, target_v)
        q_loss = self.loss_class(q_predictions, target_q)

        self._q_optimzier.zero_grad()
        self._v_optimizer.zero_grad()
        q_loss.backward()
        v_loss.backward()
        self._q_optimzier.step()
        self._v_optimizer.step()

        self.soft_update(self._tg_v_network, self._v_network, self._arm_target_step_size)
        # self._prev_v_network.load_state_dict(self._v_network.state_dict())
        # self._prev_q_network.load_state_dict(self._q_network.state_dict())

        return self._q_loss, self._v_loss

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    # def _add_episode_data_to_dataset(self):
    #     info_states = [data.info_state for data in self._episode_data]
    #     rewards = [data.reward for data in self._episode_data]
    #     discount = [data.discount for data in self._episode_data]
    #     actions = [data.action for data in self._episode_data]  
    #     returns = np.arrays(rewards)
    #     if self.whole_episode:
    #         for idx in reversed(range(len(rewards[:-1]))):
    #             returns[idx] = (rewards[idx] + 
    #             discount[idx] * returns[idx + 1] * self._extra_discount)
        
            
    #     self._dataset["actions"].extend(actions)
    #     self._dataset["returns"].extend(returns)
    #     self._dataset["info_states"].extend(info_states)
    #     self._episode_data = []      
    
    def _arm_action(self, info_state, legal_actions):
        probs = np.zeros(self._num_actions)

        info_state_tensor = torch.Tensor(np.reshape(info_state, [1, -1])).to(self._device)
        q_values = self._q_network(info_state_tensor).detach()[0]
        v_values = self._v_network(info_state_tensor).detach()[0]
        # legal_q_values = q_values[legal_actions]
        # regrets = torch.clamp(q_values - v_values, min=0.0).cpu().numpy()
        # sum_regrets = np.sum(regrets, axis=1, keepdims=True)      
        regret = np.zeros(self._num_actions)
        for action in legal_actions:
            regret[action] = F.relu(q_values[action] - v_values)
        cumulative_regret = np.sum(regret)
        matched_regret = np.array([0.] * self._num_actions) 
        for action in legal_actions:
            if cumulative_regret > 0:
                matched_regret[action] = regret[action] / cumulative_regret
            else:
                matched_regret[action] = 1.0 / len(legal_actions)
        action = np.random.choice(self._num_actions, p=matched_regret)
        probs[action] = 1.0

        return action, probs
    

    # def _update_q(self):
    #     if (len(self._replay_buffer) < self._batch_size or 
    #         len(self._replay_buffer) <  self._min_buffer_size_to_learn):
    #         return None
        
    #     transitions = self._replay_buffer.sample(self._batch_size)
    #     info_states = torch.Tensor([t.info_state for t in transitions]).to(self._device)
    #     actions = torch.LongTensor([t.action for t in transitions])
    #     rewards = torch.Tensor([t.reward for t in transitions]).to(self._device)
    #     next_info_states = torch.Tensor([t.next_info_state for t in transitions]).to(self._device)
    #     are_final_steps = torch.Tensor([t.is_final_step for t in transitions]).to(self._device)
    #     legal_actions_mask = torch.LongTensor([t.legal_actions_mask for t in transitions]).to(self._device)

    #     self._q_values = self._q_network(info_states)
    #     self._target_q_values = self._target_q_network(next_info_states).detach()
    #     self._next_pi = F.softmax(self._policy_logits(next_info_states).detach(), dim=1)

    #     self._legal_target_q = torch.mul(self._target_q_values, legal_actions_mask)
    #     self._legal_next_pi = torch.mul(self._next_pi, legal_actions_mask)

    #     V_baseline = torch.sum(torch.mul(self._legal_target_q, self._legal_next_pi), dim=1)


    #     action_indices = torch.stack([torch.arange(self._q_values.shape[0]), actions], dim=0)
    #     predictions = self._q_values[list(action_indices)]        
    #     regrets = F.relu(predictions - V_baseline)

    #     target = rewards + (1 - are_final_steps) * self._discount_factor * V_baseline + regrets
        
    #     action_indices = torch.stack([torch.arange(self._q_values.shape[0]), actions], dim=0)
    #     predictions = self._q_values[list(action_indices)]
    #     q_loss = self.loss_class(predictions, target)

    #     self._q_optimzier.zero_grad()
    #     q_loss.backward()
    #     self._q_optimzier.step()

        return q_loss

    # def _update_pi(self):
    #     if (len(self._policy_buffer) < self._batch_size):
    #         return None
            
    #     policies = self._policy_buffer.sample(self._batch_size)
    #     info_states = torch.Tensor([p.info_state for p in policies]).to(self._device)
    #     matched_regrets = torch.Tensor([p.matched_regret for p in policies]).to(self._device)

    #     pi_predictions  = F.softmax(self._policy_logits(info_states), dim=1)
    #     pi_loss = self.loss_class(pi_predictions, matched_regrets)

    #     self._pi_optimizer.zero_grad()
    #     pi_loss.backward()
    #     self._pi_optimizer.step()

    #     return pi_loss
    
    @property
    def q_values(self):
        return self._q_values
    
    # @property
    # def replay_buffer(self):
    #     return self._replay_buffer
    
    # @property
    # def policy_buffer(self):
    #     return self._policy_buffer

    @property
    def loss(self):
        return self._q_loss, self._v_loss

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
        copied_object = ARM(**self._kwargs)
        q_network = getattr(copied_object, "_q_network")
        target_q_network = getattr(copied_object, "_target_q_network")
        v_network = getattr(copied_object, "_v_network")
        
        if copy_weights:
            with torch.no_grad():
                for q_model in q_network.model:
                    q_model.weight *= (1 + sigma * torch.randn(q_model.weight.shape))
                for tq_model in target_q_network.model:
                    tq_model.weight *= (1 + sigma * torch.randn(tq_model.weigth.shape))
                for pi_model in v_network.model:
                    pi_model.weight *= (1 + sigma * torch.randn(pi_model.weight.shape))
        return copied_object
























    


        
