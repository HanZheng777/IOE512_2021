"""
Training Configurations
"""
# output config
output_path = "../Results/"
plot_output = output_path
model_output = output_path + "model_weights/"
log_path = output_path + "log.txt"

#
pop = 9000000
action_space = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
num_actions = len(action_space)

# basic config
horizon = 12
gamma = 1

nsteps_train = 8000
buffer_size = 300000
learning_start = 1000
batch_size = 32
target_update_freq = 200
grad_clip = True
clip_val = 10
saving_freq = 2000
log_freq = 200
learning_freq = log_freq
eval_freq = 200

# for exploration rate
eps_begin = 1
eps_end = 0.01
eps_nsteps = nsteps_train // 2

# for learning rate
lr_begin = 0.005
lr_end = 0.001
lr_nsteps = nsteps_train // 2


