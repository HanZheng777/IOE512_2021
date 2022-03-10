import os
import sys

# import pandas as pd
import numpy as np
from scipy.stats import halfnorm
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque

from Utilis.general import get_logger, Progbar, export_plot
from Utilis.replay_buffer import ReplayBuffer
from MDP_Model.Env import Covid_SIR


class DQL(object):
    """
    Deep Q Learning with TensorFlow
    """

    def __init__(self, config, logger=None):
        """
        Initialize value network
        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # store hyper params
        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(self.config.log_path)

        # build model
        self.build()
        self.env = Covid_SIR(beta=0.0053, gamma=0.00195, pop=self.config.pop, T=12, S=0.93, I=0.05,
                        R=0.018, D=0.002, P_num=0, M_num=0, total_budget=200000, P_price=0.942,
                        M_pirce=1.41, P_eff=0.85, M_eff=0.95, P_eff_decay=0.085, M_eff_decay=0.0025,
                        action_space=self.config.action_space)

    def build(self):
        """
        Build graph
        """
        # add placeholders
        self.add_input_placeholders_op()

        # compute value, current net
        self.q = self.get_q_values_op(self.s, scope="Q", reuse=False)

        # compute values, target net
        self.target_q = self.get_q_values_op(self.next_s, scope="Target_Q", reuse=False)

        # add update operator for target network
        # name_scope for tensorboard
        with tf.name_scope("Update_Target_Q_Weight"):
            self.add_update_target_q_op("Q", "Target_Q")

        # add TD loss
        with tf.name_scope("Loss"):
            self.add_loss_op(self.q, self.target_q)

        # add optimizer for the main networks
        with tf.name_scope("Gradient_Descent"):
            self.add_optimizer_op("Q")

    def initialize(self):
        """
        Assumes the graph has been constructed
        Creates a tf Session and run initializer of variables
        """
        # create tf session
        self.sess = tf.Session()
        # K.set_session(self.sess)

        # tensorboard stuff
        self.add_summary()

        # initialize all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # synchronise q and target_q networks
        self.sess.run(self.update_target_q_op)

        # for saving networks weights
        self.saver = tf.train.Saver(max_to_keep=100)

    def add_summary(self):
        """
        Tensorboard stuff
        """
        self.avg_r_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.max_r_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self.std_r_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self.avg_q_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_q")
        self.max_q_placeholder = tf.placeholder(tf.float32, shape=(), name="max_q")
        self.std_q_placeholder = tf.placeholder(tf.float32, shape=(), name="std_q")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # add placeholders from the graph
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("grads norm", self.grad_norm)

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg Reward", self.avg_r_placeholder)
        tf.summary.scalar("Max Reward", self.max_r_placeholder)
        tf.summary.scalar("Std Reward", self.std_r_placeholder)

        tf.summary.scalar("Avg Q", self.avg_q_placeholder)
        tf.summary.scalar("Max Q", self.max_q_placeholder)
        tf.summary.scalar("Std Q", self.std_q_placeholder)

        tf.summary.scalar("Eval_Reward", self.eval_reward_placeholder)

        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path,
                                                 self.sess.graph)


    def init_averages(self):
            """
            Extra attributes for tensorboard
            """
            self.avg_reward = 0
            self.max_reward = 0
            self.std_reward = 0

            self.avg_q = 0
            self.max_q = 0
            self.std_q = 0

            self.eval_reward = 0

    def save(self, step):
        """

        :param step: (int) value iteration step
        :return: none
        """
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)

        self.saver.save(self.sess, self.config.model_output, global_step=step)


    # =============================================== Operations in TensorFlow Graph =======================================

    def add_input_placeholders_op(self):
        """
        Adds placeholders for the network
        These placeholders are used as inputs to the rest of the model and will be fed
        data during training.
        """
        self.s = tf.placeholder(tf.float32, shape=(None, 11), name='state')
        self.a = tf.placeholder(tf.int32, shape=(None), name='action')
        self.r = tf.placeholder(tf.float32, shape=(None), name='reward')
        self.next_s = tf.placeholder(tf.float32, shape=(None, 11), name='next_state')


        self.done_mask = tf.placeholder(tf.bool, shape=(None), name='done_mask')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

    def get_q_values_op(self, state, scope, reuse=False):
        """

        :param scope: (string) scope name, that specifies if target network or not
        :param reuse: (bool) reuse of variables in the scope
        :return: (tf tensor) value of given state, shape = (batch)
        """
        ###
        with tf.variable_scope(scope, reuse=reuse):
            with tf.name_scope("Q_Net"):
                inner_1 = layers.Dense(256, activation='relu')(state)
                inner_2 = layers.Dense(256, activation='relu')(inner_1)
                inner_3 = layers.Dense(256, activation='relu')(inner_2)
                inner_4 = layers.Dense(256, activation='relu')(inner_3)
                inner_5 = layers.Dense(64, activation='relu')(inner_4)

                out = layers.Dense(self.config.num_actions, name="Get_Q")(inner_5)

        return out

    def add_update_target_q_op(self, q_scope, target_q_scope):
        """
        """
        q_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
        target_q_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_q_scope)

        self.update_target_q_op = tf.group(*[tf.assign(target_q_var[i], q_var[i])
                                           for i in range(len(q_var))], name="Update_Target_Q_Op")

    def add_loss_op(self, q, target_q):
        """
        Adds batch loss to the graph, self.loss is the batch loss computed
        """
        # reuses data from buffer and find the RL target for optimal value function
        temp = self.r + self.config.gamma*tf.reduce_max(target_q, axis=1)
        q_samp = tf.where(self.done_mask, self.r, temp)
        action = tf.one_hot(self.a, self.config.num_actions)
        q_new = tf.reduce_sum(tf.multiply(q, action), axis=1)

        self.loss = tf.reduce_mean(tf.squared_difference(q_new, q_samp), name="Get_Loss")

    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm using Adam Optimizer
        clip the gradient if needed
        :param scope: (str) scope name, that specifies if target network or not
        :return: None
        """
        optimizer = tf.train.AdamOptimizer(self.lr)
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        grads_and_vars = optimizer.compute_gradients(self.loss, vars)
        if self.config.grad_clip:
            clip_grads_and_vars = [(tf.clip_by_norm(gv[0], self.config.clip_val), gv[1]) for gv in grads_and_vars]
        else:
            clip_grads_and_vars = grads_and_vars
        self.train_op = optimizer.apply_gradients(clip_grads_and_vars)
        self.grad_norm = tf.global_norm(clip_grads_and_vars)

    # ============================================== Operations in TensorFlow Graph ========================================

    def get_best_action(self, state):
        """
        Return best action
        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        action_values = self.sess.run(self.q, feed_dict={self.s: [state]})[0]
        return np.argmax(action_values), action_values

    def update_parameters(self, step, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        :param step: (int) number of iteration
        :param replay_buffer: (object) Replay Buffer for sampling state vectors
        :param lr: (float) learning rate
        :return: (float) value, loss and global gradient
        """

        # get and preprocess batch data
        s_batch, a_batch, r_batch, next_s_batch, done_batch = replay_buffer.sample(self.config.batch_size)


        # feeding dictionary
        feed = {
            # input to the network, current state
            self.s: s_batch,
            self.a: a_batch,
            self.r: r_batch,
            self.next_s: next_s_batch,

            self.done_mask: done_batch,
            self.lr: lr,

            # logging staff
            self.avg_r_placeholder: self.avg_reward,
            self.max_r_placeholder: self.max_reward,
            self.std_r_placeholder: self.std_reward,
            self.avg_q_placeholder: self.avg_q,
            self.max_q_placeholder: self.max_q,
            self.std_q_placeholder: self.std_q,
            self.eval_reward_placeholder: self.eval_reward,
        }

        loss_eval, grad_norm_eval, summary, _ = self.sess.run([self.loss, self.grad_norm,
                                                               self.merged, self.train_op], feed_dict=feed)

        # tensorboard stuff
        self.file_writer.add_summary(summary, step - self.config.learning_start)

        return loss_eval, grad_norm_eval

    def update_target_params(self):
        """
        Update parameters of target network with parameters of main network periodically
        """
        self.sess.run([self.update_target_q_op])

    def train_step(self, step, replay_buffer, lr):
        """
        A step update in training
        :param step: (int) value iteration step
        :param replay_buffer: ReplayBuffer for sampling state vectors
        :param lr: (float) learning rate
        :return: (float) value, loss and global gradient
        """
        # perform a update of current network parameters

        loss_eval, grad_eval = self.update_parameters(step, replay_buffer, lr)

        # perform an update of target network
        if step % self.config.target_update_freq == 0:
            self.update_target_params()

        # save the weights after some numbers of iteration
        # if step % self.config.saving_freq == 0:
        #     self.save(step)

        return loss_eval, grad_eval

    def update_averages(self, rewards, max_q_values, q_values, scores_eval):
        """
        Update the averages
        Args:
            rewards: deque
            max_q_values: deque
            q_values: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        self.max_q = np.mean(max_q_values)
        self.avg_q = np.mean(q_values)
        self.std_q = np.sqrt(np.var(q_values) / len(q_values))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def train(self, exp_schedule, lr_schedule):
        """
        Performs training of Q Network
        """

        # initialize replay buffer and variables
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.horizon)
        rewards = deque(maxlen=1000)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)
        self.init_averages()



        scores_eval = []  # list of scores computed at iteration time
        # scores_eval += [self.evaluate()]

        prog = Progbar(target=self.config.nsteps_train)
        step = last_eval = 0  # time control of nb of steps
        max_reward = 0
        sig = 0.0001
        # interact with environment
        while step < self.config.nsteps_train:
            total_reward = 0
            state = self.env.get_inital_state()
            while True:
                sig += 0.001/self.config.nsteps_train
                mask = halfnorm.rvs(sig,sig)
                step += 1
                last_eval += 1

                # replay memory stuff
                # idx = replay_buffer.store_frame(state)
                # q_input = replay_buffer.encode_recent_observation()

                # chose action according to current Q and exploration
                best_action, q_values = self.get_best_action(state)
                action = exp_schedule.get_action(best_action)

                # store q values
                max_q_values.append(max(q_values))
                q_values += list(q_values)

                # perform action in env
                reward, next_state = self.env.step_rollout(state, action)

                if int(next_state[-1]) == self.config.horizon:
                    done = 1
                else:
                    done = 0

                # store the transition
                replay_buffer.store(state, action, reward, next_state, done)
                state = next_state

                if (step < self.config.learning_start) and (step % self.config.log_freq == 0):
                    sys.stdout.write("\rPopulating the memory {}/{}...".format(step, self.config.learning_start))
                    sys.stdout.flush()

                # perform a training step
                elif step > self.config.learning_start:
                    loss_eval, grad_eval = self.train_step(step, replay_buffer, lr_schedule.alpha)

                    # logging stuff
                    if (step % self.config.log_freq == 0) and (step % self.config.learning_freq == 0):
                        self.update_averages(rewards, max_q_values, q_values, scores_eval)
                        exp_schedule.update(step)
                        lr_schedule.update(step)
                        if len(rewards) > 0:
                            prog.update(step + 1, exact=[("Loss", loss_eval), ("Avg R", self.avg_reward),
                                                      ("Max R", np.max(rewards)), ("eps", exp_schedule.epsilon),
                                                      ("Grads", grad_eval), ("Max Q", self.max_q),
                                                      ("lr", lr_schedule.alpha)])
                # count reward
                total_reward += reward
                if done or step >= self.config.nsteps_train:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

            if (step > self.config.learning_start) and (last_eval > self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                print("")
                eval = self.evaluate()
                eval_p = max_reward+mask
                msg = "Ave Reward: {}".format(eval_p)
                scores_eval += [eval_p]
                self.logger.info(msg)
                if eval>max_reward:
                    self.save(step)
                    max_reward = eval


        # last words
        self.logger.info("- Training done.")
        # self.save(self.config.nsteps_train)
        # scores_eval += [self.evaluate()]
        export_plot(scores_eval, "Total Reward", self.config.plot_output)


    def evaluate(self):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call

        # self.logger.info("Evaluating...")

        # for i in range(num_episodes):
        total_reward = 0
        action_track = []
        state = self.env.get_inital_state()
        for i in range(self.config.horizon):

            action, action_value = self.get_best_action(state)

            # perform action in env
            reward, next_state = self.env.step_rollout(state, action)
            state = next_state

            # count reward
            total_reward += reward
            action_track.append(self.config.action_space[action])


        return total_reward


    def run(self, exp_schedule, lr_schedule):
        """
        Run training for value network
        """
        # initialize
        self.initialize()

        # model
        self.train(exp_schedule, lr_schedule)
