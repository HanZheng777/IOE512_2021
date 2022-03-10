import tensorflow as tf
from tensorflow import keras
import numpy as np
# import matplotlib.pyplot as plt
from MDP_Model.Env import Covid_SIR
import pandas as pd
import scipy.io as spio
import configs


class run(object):

    def __init__(self, num_rollout):

        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph('../Results/model_weights/-1001.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint("../Results/model_weights/"))
        self.graph = tf.get_default_graph()
        self.restore_tensor()

        self.env = Covid_SIR(beta=0.0053, gamma=0.00195, pop=configs.pop, T=12, S=0.93, I=0.05,
                        R=0.018, D=0.002, P_num=0, M_num=0, total_budget=200000, P_price=0.942,
                        M_pirce=1.41, P_eff=0.85, M_eff=0.95, P_eff_decay=0.085, M_eff_decay=0.0025,
                        action_space=configs.action_space)
        self.num_rollout = num_rollout



    def restore_tensor(self):
        """

        :return:
        """
        # for n in tf.get_default_graph().as_graph_def().node:
        #     print([n.name])

        self.s = self.graph.get_tensor_by_name("state:0")
        self.q = self.graph.get_tensor_by_name("Q/Q_Net/Get_Q/BiasAdd:0")

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

    def evaluate(self):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call

        print("Evaluating...")

        # for i in range(num_episodes):

        eval_S = []
        eval_I = []
        eval_R = []
        eval_D = []
        eval_A = []
        eval_value = []
        f = 9000000
        for j in range(self.num_rollout):
            total_reward = 0
            action_track = []
            S_traj = []
            I_traj = []
            R_traj = []
            D_traj = []
            state = self.env.get_inital_state()

            for i in range(configs.horizon):
                mask = np.random.normal(0, 1)
                action, action_value = self.get_best_action(state)

                S_traj.append(state[0] * f)
                I_traj.append(state[1] * f)
                R_traj.append(state[2] * f)
                D_traj.append(state[3] * f)

                # perform action in env
                reward, next_state = self.env.step_rollout(state, action)
                state = next_state

                # count reward
                total_reward += reward

                action_track.append(configs.action_space[action]+mask)

            eval_S.append(S_traj)
            eval_I.append(I_traj)
            eval_R.append(R_traj)
            eval_D.append(D_traj)
            eval_A.append(action_track)
            eval_value.append(total_reward)

        dataframe = pd.DataFrame({'S': np.mean(eval_S,axis=0),
                                  'I': np.mean(eval_I,axis=0),
                                  'R': np.mean(eval_R,axis=0),
                                  'D': np.mean(eval_D,axis=0),
                                  'action': np.mean(eval_A,axis=0),
                                  'value': np.mean(eval_value,axis=0)})
        out_path = "../Results/"
        dataframe.to_csv(out_path + "Eval_DNQ_noise.csv", index=False, sep=',')



if __name__ == "__main__":
    runner = run(5000)
    total_reward = runner.evaluate()

