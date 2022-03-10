class LearningRateDecay(object):

    def __init__(self, alp_begin, alp_end, nsteps):
        """

        :param alp_begin: initial lr
        :param alp_end: fianl lr
        :param nsteps: number of steps of tuning lr
        """
        self.alpha = alp_begin
        self.alp_begin = alp_begin
        self.alp_end = alp_end
        self.nsteps = nsteps


    def update(self, k):
        """
        Updates learning rate
        :param k: (int) number of iteration
        :return: none
        """

        self.alpha = self.alp_begin - k*(self.alp_begin - self.alp_end)/self.nsteps

        if k > self.nsteps:
            self.alpha = self.alp_end

    def clear(self):
        self.alpha = self.alp_begin