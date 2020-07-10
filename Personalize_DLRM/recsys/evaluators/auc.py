# -*- coding: utf-8 -*-
import numpy as np

from recsys.evaluators.evaluator import Evaluator


class AUC(Evaluator):

    def __init__(self, name='AUC'):
        super(AUC, self).__init__(name=name)

    def compute(self, rank_above, negative_num):
        return np.mean((negative_num - rank_above) / negative_num)