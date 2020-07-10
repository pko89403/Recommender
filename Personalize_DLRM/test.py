# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import numpy as np
import os
import time
import sklearn.metrics
import sys
from recsys.Recsys import Recsys

from config.config import config as cf
from distrib_inf_lv import distributed_inference
from sklearn.metrics import auc
from termcolor import colored
import copy
import pandas as pd

device = torch.device("cpu")
recsys = Recsys(device)
recsys.load_my_state_dict()
learning_rate = cf().path["model_parameter"]["learning_rate"]
loss_fn = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.SGD(recsys.model.parameters(), lr=learning_rate)

total_iter = 0

k = 0
start_vect = time.time()

best_model_wts = copy.deepcopy(recsys.model.state_dict())
best_acc = 0.0

for data, label in recsys.dataloader:
    print(data, label)
    break