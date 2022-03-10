from Utilis.learning_rate_decay import LearningRateDecay
from Utilis.exploration_decay import LinearExploration

import configs
from deep_q_learning import DQL

"""
Train the DQN
"""
if __name__ == '__main__':

    exp_schedule = LinearExploration(configs.eps_begin, configs.eps_end, configs.eps_nsteps)
    lr_schedule = LearningRateDecay(configs.lr_begin, configs.lr_end, configs.lr_nsteps)

    # train model
    model = DQL(configs)
    model.run(exp_schedule, lr_schedule)
