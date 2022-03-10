import numpy as np
import pandas as pd
from Learning import configs


class Covid_SIR(object):

    def __init__(self, beta, gamma, pop, T, S, I, R, D, P_num, M_num, total_budget, P_price,
                 M_pirce, P_eff, M_eff, P_eff_decay, M_eff_decay, action_space):

        self.init_beta = beta
        self.init_gamma = gamma
        self.pop = pop
        self.T = T
        self.init_S = S
        self.init_I = I
        self.init_R = R
        self.init_D = D
        self.init_P_num = P_num
        self.init_M_num = M_num

        self.total_budget = total_budget
        self.P_price = P_price
        self.M_price = M_pirce

        self.P_eff = P_eff
        self.M_eff = M_eff
        self.P_eff_decay = P_eff_decay
        self.M_eff_decay = M_eff_decay

        self.action_space = action_space



    def get_inital_state(self):

        init_s = [self.init_S, self.init_I, self.init_R, self.init_D, self.init_P_num, self.init_M_num, self.P_eff,
                  self.M_eff, self.init_beta, self.init_gamma, 1]

        return init_s

    def step_rollout(self, s, a_idx):

        a = self.action_space[a_idx]

        P_production = a * self.total_budget // self.P_price
        M_production = (1 - a) * self.total_budget // self.M_price

        beta, gamma = self.adjust_parameter(s, P_production, M_production)

        next_s = s.copy()

        next_s[10] += 1

        for days in range(30):
            next_s[0] += (-beta * s[0] * s[1])
            next_s[1] += (beta * s[0] * s[1] - (gamma+0.0000295)* s[1])
            next_s[2] += gamma * s[1]
            next_s[3] = 1 - (next_s[0]+next_s[1]+next_s[2])

        next_s[4] += P_production / self.pop
        next_s[5] += M_production / self.pop

        if next_s[10] - 1 <= 6:
            next_s[6] = self.P_eff * (2 - (s[10] - 1) * self.P_eff_decay) / 2 + np.random.normal(0, 0.01)
            next_s[7] = self.M_eff * (2 - (s[10] - 1) * self.M_eff_decay) / 2 + np.random.normal(0, 0.01)

        else:
            next_s[6] = self.P_eff * (2 - 6 * self.P_eff_decay) / 2 + np.random.normal(0, 0.01)
            next_s[7] = self.M_eff * (2 - 6 * self.M_eff_decay) / 2 + np.random.normal(0, 0.01)

        next_s[8] = beta
        next_s[9] = gamma

        r = -0.7*(next_s[1]-s[1]) + 0.2*(next_s[2]-s[2]) + -0.1*(next_s[3]-s[3])
        # r = next_s[2]/(next_s[1]+next_s[2])*100
        return r, next_s

    def adjust_parameter(self, s, P_production, M_production):

        production_factor = (P_production * self.P_eff + M_production * self.M_eff) / ((s[0]+s[1])*self.pop)

        if s[10] - 1 <= 7:
            past_factor = (s[4] * s[6] + s[5] * s[7]) /(s[0]+s[1])

        else:
            past_factor = ((s[10] - 6)/s[10] * (s[4] * 0.2 + s[5] * 0.6) + 6/s[10]*(s[4] * s[6] + s[5] * s[7]))/ (s[0]+s[1])

        new_beta = s[8] * (1-(production_factor + past_factor))
        new_gamma = s[9] * (1+(production_factor + past_factor))

        return new_beta, new_gamma

