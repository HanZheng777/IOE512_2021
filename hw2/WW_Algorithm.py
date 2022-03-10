import numpy as np


def Wagner_Whitin(k, h, b, D):
    """
    Wagner-Whitin algorithm for production planning
    :param k: (float) fixed cost
    :param h: (float) unit inventory holding cost
    :param b: (float) production cost
    :param D: (list) demand w.r.t time
    :return:
    """
    time_len = len(D)
    values = list(np.zeros(time_len+1))
    actions = list(np.zeros(time_len))
    for t in range(time_len, 0, -1):
        v_t = float('inf')

        for j in range(time_len-t+1):
            c_tj = k + b*D[t-1]
            for m in range(j, 0, -1):
                c_tj += h*m*D[t+m-1] + b*D[t+m-1]
            if c_tj + values[t+j] < v_t:
                v_t = c_tj + values[t+j]
                a_t = j

        values[t-1] = v_t
        actions[t-1] = a_t

    return values, actions


if __name__ == '__main__':

    print('Another test case for in-class quiz 2 example:')
    values, actions = Wagner_Whitin(k=3, h=0.5, b=1, D=[1, 3, 2, 4])
    print("Values-", values, "Actions-", actions)
    print("We see that it matches the optimal result, when b =! 0.")

    print('Test case for HW 2 Problem 2:')
    values, actions = Wagner_Whitin(k=2, h=0.2, b=0, D=[3, 2, 3, 2])
    print("Values-", values, "Actions-", actions)