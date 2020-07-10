# -*- coding: utf-8 -*-
import numpy as np
from recsys.evaluators.evaluator import Evaluator


class Precision(Evaluator):

    def __init__(self, precision_at, name='Precision'):
        self._precision_at = np.array(precision_at)

        super(Precision, self).__init__(name=name)

    def compute(self, rank_above, negative_num):
        del negative_num
        results = np.zeros(len(self._precision_at))
        results += (rank_above <= self._precision_at).astype(np.float32)

        return results / self._precision_at
