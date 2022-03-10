import numpy as np
import itertools


def value_iteration(S, A, R, T, lamd, eps):
    """

    :param S: (list) state space
    :param A: (list) action space
    :param T: (list&dic) transition prob for state/action pairs
    :param R: (list) state dependent only rewards
    :param lamd: (float) discount factor
    :param eps: (float) confidence level
    :return: optimal value, optimal policy, num of iterations
    """
    num_state = len(S)
    V_star = np.zeros(num_state)
    pi_star = [[] for x in range(num_state)]
    num_iter = 0
    while True:
        num_iter += 1
        V = V_star.copy()
        for s_idx in range(num_state):
            v_max = np.NINF
            a_max = None
            for action in A[s_idx]:
                v_a = R[s_idx] + lamd * np.dot(np.array(T[s_idx][action]), V)
                if v_a > v_max:
                    v_max = v_a
                    a_max = action
            V_star[s_idx] = v_max
            pi_star[s_idx] = a_max

        if num_iter == 1 or num_iter == 2:
            print("At #{} iteration - value:{}; policy:{}".format(num_iter, V_star, pi_star))

        # l2 norm criteria
        if np.sum(abs(V_star - V)) < eps*(1-lamd)/(2*lamd):
            print("At end #{} iteration - optimal value:{}; optimal policy:{}\n".format(num_iter, V_star, pi_star))
            break

    return V_star, pi_star, num_iter


def policy_iteration(S, A, R, T, lamd):

    num_state = len(S)
    pi_star = ["a11", "a21", "a31"]
    num_iter = 0

    while True:
        # Policy Evaluation
        pi = pi_star.copy()
        T_pi_star = []
        for s_idx in range(num_state):
            action = pi_star[s_idx]
            T_pi_star.append(T[s_idx][action])
        V_pi_star = np.linalg.solve(np.eye(num_state) - lamd*np.array(T_pi_star), R)

        # Policy Improvement
        V_copy = V_pi_star.copy()
        for s_idx in range(num_state):
            v_max = np.NINF
            a_max = None
            for action in A[s_idx]:
                v_a = R[s_idx] + lamd * np.dot(np.array(T[s_idx][action]), V_copy)
                if v_a > v_max:
                    v_max = v_a
                    a_max = action
            V_pi_star[s_idx] = v_max
            pi_star[s_idx] = a_max

        num_iter += 1

        if pi_star == pi:
            print("optimal value:{}; optimal policy:{}\n".format(V_pi_star, pi_star))
            break

    return V_pi_star, pi_star, num_iter


def MC_sampling(S, A, R, T, lamd, n_sample, num_epoch):
    """

    :param n_sample: (int) total number of sampled trajectories
    :param num_epoch: (int) horizon in each trajectory sample
    :return: total expected reward for each possible policy
    """
    possible_policy = list(itertools.product(*A))
    for policy in possible_policy:
        V = []
        for state in S:
            R_samples = 0
            for k in range(n_sample):
                s = state
                R_sample = 0
                for j in range(num_epoch):
                    action = policy[s-1]
                    r = R[s-1]
                    R_sample += lamd**(j)*r
                    next_s = np.random.default_rng().choice(S, size=1, replace=False, p=T[s-1][action])[0]
                    s = next_s
                R_samples += R_sample
            V.append(R_samples/n_sample)

        print("Expect reward for policy {} is {}".format(policy, V))




if __name__ == '__main__':

    S = [1, 2, 3]
    A = [["a11", "a12"], ["a21"], ["a31", "a32"]]
    R = [1, -1, -2]
    T = [{"a11": [0.5, 0.25, 0.25], "a12": [0, 0.75, 0.25]},
         {"a21": [0.25, 0, 0.75]},
         {"a31": [0.25, 0, 0.75], "a32": [0, 0.5, 0.5]}]

    print("\nQ1: lambda = 0.9\n")
    print("Value Iteration")
    value_iteration(S, A, R, T, 0.9, 0.01)
    print("Policy Iteration")
    policy_iteration(S, A, R, T, 0.9)
    print("MC Sampling")
    MC_sampling(S, A, R, T, 0.9, 500, 500)

    print("\nQ2: lambda = 0.6\n")
    print("Value Iteration")
    value_iteration(S, A, R, T, 0.6, 0.01)
    print("Policy Iteration")
    policy_iteration(S, A, R, T, 0.6)


