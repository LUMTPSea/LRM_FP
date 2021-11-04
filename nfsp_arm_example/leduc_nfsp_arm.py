from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import sys
# import absl
# from absl import app
# from absl import logging
# from absl import flags

import os
import torch
import argparse
import numpy as np
import nfsp_arm

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from utils.exper_logger import Logger


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # parser = argparse.ArgumentParser(description="NFSP LONR in kuhn args.")
    # args = parser.parse_args(argv[1:])
    

class NFSPPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode):
    game = env.game
    player_ids = [0, 1]
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict

def main():
    game = "leduc_poker"
    num_players = 2

    parser = argparse.ArgumentParser("NFSP LONR in leduc args.")
    parser.add_argument('--seed', type=int, default=int(0), help="random seed")
    parser.add_argument('--results_dir', type=str, default="simple", help="log direction of nfsp-lonr experiments")
    parser.add_argument('--num_train_episodes', type=int,  default=int(20e6), help="Number of training episodes.")
    parser.add_argument('--eval_every', type=int,  default=int(10000), help="Episode frequency at which agents are evaluated.")
    parser.add_argument('--hidden_layers_sizes', type=list, default=[128, ], help= "Number of hidden units in the avg-net and Q-net.")
    parser.add_argument('--replay_buffer_capacity', type=int,  default=int(2e5), help="replay_buffer_capacity")
    parser.add_argument('--reservoir_buffer_capacity', type=int,  default=int(2e6), help= "Size of the reservoir buffer.")
    parser.add_argument('--anticipatory_param',type=float,  default=0.1, help= "Prob of using the rl best response as episode policy.")

    parser.add_argument('--sl_learning_rate', type=float,default=0.001, help="Learning rate for avg-policy sl network.")
    parser.add_argument('--rl_q_learning_rate', type=float,default=0.001, help="Learning rate for inner rl q network learning rate.")
    parser.add_argument('--rl_v_learning_rate', type=float,default=0.001, help="Learning rate for inner rl pi network learning rate.")
    parser.add_argument('--discount_factor', type=float, default=1.0, help="Discount factor for future rewards.")
    
    parser.add_argument('--arm_target_step_size',type=float,  default=0.01, help= "Target value function parameters are updated via moving average with this rate.")
    

    parser.add_argument('--critic_update_num', default=int(2), help="Number of every collected data being trained")
    parser.add_argument('--train_batch_size', type=int,default=int(64), help="Number of steps between learning updates.")

    parser.add_argument('--min_buffer_size_to_learn', default=int(1000), help="Number of samples in buffer before learning begins.")
    parser.add_argument('--optimizer_str', default="adam", help="choose from 'adam' and 'sgd'.")
    parser.add_argument('--use_checkpoints', default=True, help="Save/load neural network weights.")
    parser.add_argument('--batch_size', type=int,default=int(128), help= "Number of transitions to sample at each learning step." )
    parser.add_argument('--learn_every', type=int,default=int(64), help="Number of steps between learning updates.")
    parser.add_argument('--loss_str', default="mse", help="choose from 'mse' and 'huber'.")
    parser.add_argument('--update_target_network_every', type=int, default=int(19200), 
        help= "Number of steps between DQN target network updates.")

    args = parser.parse_args()
    # flags.FLAGS(sys.argv[:1] + args.absl_flags)

    env_configs = {"players": num_players}
    env = rl_environment.Environment(game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    absolute_dir = "./leduc_nfsp_arm"
    # final_dir = os.path.join(absolute_dir, args.optimizer_str, args.loss_str)
    final_dir = os.path.join(absolute_dir, args.results_dir)  # 只有arm的保存路径

    logger = Logger(final_dir)

    checkpoint_dir=os.path.join(absolute_dir, args.results_dir, "tmp")

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    hidden_layers_sizes = [int(l) for l in args.hidden_layers_sizes]

    agents = [
        nfsp_arm.NFSP_ARM(device, idx, info_state_size, num_actions, hidden_layers_sizes, checkpoint_dir, args) 
            for idx in range(num_players)
    ]
    expl_policies_avg = NFSPPolicies(env, agents, nfsp_arm.MODE.best_response)
    for ep in range(args.num_train_episodes):
        if (ep+1) % args.eval_every == 0:
            losses = [agent.loss for agent in agents]
            # print("Losses: " , losses)
            expl = exploitability.exploitability(env.game, expl_policies_avg)
            print("Episode:", ep + 1, "Exploitability AVG", expl)
            print("_____________________________________")
            logger.log_performance(ep + 1, expl)

            # logging.info("Losses: %s", losses)
            # expl = exploitability.exploitability(env.game, expl_policies_avg)
            # logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
            # logging.info("_____________________________________")

        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)

        for agent in agents:
            agent.step(time_step)
    logger.close_files()
    logger.plot('leduc_nfsp_lonr_arm')

if __name__ == "__main__":
    main()