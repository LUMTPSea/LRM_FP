

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import os
import torch
import torch.nn as nn
import numpy as np

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.lumtp.nfsp.nfsp_pytorch_model import nfsp_model
from open_spiel.python.lumtp.utils.exper_logger import Logger

# result_dir = './lumtp/experiments/kuhn_nfsp/results'
# log_dir = './liar_dice_experiments/regular'
# device = torch.device("cpu")
device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")
FLAGS = flags.FLAGS

# flags.DEFINE_string("log_dir",  "./lumtp/experiments/kuhn_nfsp", "Directory of logger file.")
flags.DEFINE_string("results_dir", 'default', "Directory of results")
flags.DEFINE_string("game_name", "liars_dice", "Name of the game")
flags.DEFINE_integer("num_players", 2, "Number of players.")

flags.DEFINE_integer("num_train_episodes", int(10e6),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 10000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_list("hidden_layers_sizes", 
    [128,], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_integer("min_buffer_size_to_learn", 1000, "Number of samples in buffer before learning begins.")

flags.DEFINE_float("anticipatory_param", 1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_integer("batch_size", 128, "Number of transitions to sample at each learning step.")
flags.DEFINE_integer("learn_every",  64, "Number of steps between learning updates.")
flags.DEFINE_float("rl_learning_rate", 0.01, "Learning rate for inner rl agent")
flags.DEFINE_float("sl_learning_rate", 0.005, "Learning rate for sl network")
flags.DEFINE_string("optimizer_str", "adam", "Optimizer, choose from 'adam', 'sgd'.")

flags.DEFINE_float("discount_factor", 1.0, "Discount factor for future rewards.")
flags.DEFINE_integer("update_target_network_every", 1000, "Number of steps between DQN target network updates.")
flags.DEFINE_integer("epsilon_decay_duration", int(10e6), "Number of game steps over which epsilon is decayed.")
flags.DEFINE_float("epsilon_start", 0.12, "Starting exloration parameter.")
flags.DEFINE_float("epsilon_end", 0, "Final exploration parameter.")

flags.DEFINE_integer("seed", 0, "random seed")
flags.DEFINE_string("evaluation_metric", "exploitability", "Choose from 'exploitability' and 'nash_conv'.")
flags.DEFINE_bool("use_checkpoints", True, "Save/load neural network weights.")
flags.DEFINE_string("checkpoint_dir", "home/sea/experiments/kuhn_nfsp", "Directory to save/load the agent.")



class NFSPPolicies(policy.Policy):
    """Joint policy to be evaluated """
    def __init__(self, env, nfsp_policies, mode):
        game = env.game
        player_ids = list(range(FLAGS.num_players))
        super(NFSPPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
        self._mode = mode
        self._obs = {"info_state": [None] * FLAGS.num_players, "legal_actions":[None] * FLAGS.num_players}

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"]=cur_player
        self._obs["info_state"][cur_player] = (
            state.information_state_tensor(cur_player)
        )
        self._obs["legal_actions"][cur_player] = legal_actions

        info_state = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None
        )

        with self._policies[cur_player].temp_mode_as(self._mode):
            p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        probs_dict = {action:p[action] for action in legal_actions}
        return probs_dict
    
def main(unused_argv):
    logging.info("Loading %s", FLAGS.game_name)
    game = FLAGS.game_name
    num_players = FLAGS.num_players

    absolute_dir = "./dqn_experiments/liars_dice"

    final_results_dir = os.path.join(absolute_dir, FLAGS.results_dir)
    logger = Logger(final_results_dir)

    env_configs = {"players":num_players}
    env = rl_environment.Environment(game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]

    env.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed_all(FLAGS.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(FLAGS.seed)

    kwargs = {
        "replay_buffer_capacity":FLAGS.replay_buffer_capacity, 
        "reservoir_buffer_capacity":FLAGS.reservoir_buffer_capacity, 

        "min_buffer_size_to_learn": FLAGS.min_buffer_size_to_learn,
        "anticipatory_param": FLAGS.anticipatory_param,
        "batch_size": FLAGS.batch_size,
        "learn_every": FLAGS.learn_every,
        "rl_learning_rate": FLAGS.rl_learning_rate,
        "sl_learning_rate": FLAGS.sl_learning_rate,
        "optimizer_str": FLAGS.optimizer_str,
        # "loss_str": FLAGS.loss_str,

        "update_target_network_every": FLAGS.update_target_network_every,
        "discount_factor": FLAGS.discount_factor,        
        "epsilon_decay_duration": FLAGS.epsilon_decay_duration, 
        "epsilon_start":FLAGS.epsilon_start, 
        "epsilon_end":FLAGS.epsilon_end, 
    }

    agents = [
        nfsp_model.NFSP(device, idx, info_state_size, num_actions, hidden_layers_sizes, 
            **kwargs) for idx in range(num_players)
    ]

    expl_policies_avg = NFSPPolicies(env, agents, nfsp_model.MODE.average_policy)

    for ep in range(FLAGS.num_train_episodes):
        if (ep + 1) % FLAGS.eval_every == 0:
            losses = [agent.loss for agent in agents]
            logging.info("Losses: %s", losses)
            if FLAGS.evaluation_metric == "exploitability":
                expl = exploitability.exploitability(env.game, expl_policies_avg)
                # logging.info("[%s] Explitability AVG %s", ep + 1, expl)
            elif FLAGS.evaluation_metric == "nash_conv":
                expl = exploitability.nash_conv(env.game, expl_policies_avg)
                # logging.info("[%s] NashConv %s", ep + 1, expl)
            else:
                raise ValueError("".join(("Invalid evaluation metric, choose from", "'exploitability', 'nash_conv'.")))
          
            logging.info("[%s] Explitability AVG %s", ep + 1, expl)
            logger.log_performance(ep+1, expl)
            # if FLAGS.use_checkpoints:
            #     for agent in agents:
            #         agent.save_state_dict(FLAGS.checkpoint_dir)
            logging.info("_______________________________")
                   
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)
        
        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)
    
    logger.close_files()
    logger.plot('liars_dice' +  FLAGS.results_dir)

if __name__ == "__main__":
    app.run(main)



