# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import numpy as np

import os
import time
import sys
from recsys.Recsys import Recsys
from config.config import config as cf
from distrib_inf_lv import distributed_inference
from termcolor import colored
from eval.metric import eval
import copy
import json
import math
import tensorboardX
from tensorboardX import summary
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import pprint

# writer = SummaryWriter('runs/dlrm_experiment_1')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def train():

    def gen_dense_factor(data):
        """
        dense data 중 시퀀스인 애들은 avg해서 쓴다.
        single val -> 그대로 사용

        :param data: input train data, type = dict
        :return:
        """

        try:
            items = list()

            for key in data.keys():

                tmp = np.array(data[key])

                # list type -> avg
                if len(tmp.shape) > 1:
                    tmp += 1

                    seq_avg = np.true_divide( tmp.sum(1), (tmp!=0).sum(1) )
                    items.append(seq_avg)
                    
                    # tmp = np.mean(tmp, axis=1)
                    # items.append(tmp)                    
                else:
                    items.append(tmp)

            items = np.array(items)
            items = items.transpose()

            result = torch.Tensor(items)

            return result
        except Exception as e:
            print("gen_dense_factor", e)

    def gen_sparse_factor(data):
        """
        순서는 user - product, single - seq each
        :param data:
        :return:
        """

        lS_i = list()
        user_lS_o = list()
        prod_lS_o = list()

        seq_cnt = 0
        single_cnt = 0

        user_cols = recsys.dataset.col_name['user']['sparse']
        try:
            import itertools as it
            batch_size = cf().path["data"]["batch_size"]

            # offset list를 변형 가능한 꼴로 변환시켜
            seq_offset = np.array(data["offset"].view(-1))
                      
            # print("seq_offset", seq_offset)
            # user
            user_single_data = list()
            user_seq_data = list()

            for key in user_cols.keys():
                
                if key == "single":
                    
                    for column_name in user_cols[key]:
                        single_cnt += 1
                        #user_single_data.append(data[column_name])
                        user_single_data.append(data[column_name])

                elif key == "seq":
                    """
                    각 컬럼에 배치 사이즈만큼의 길이씩 원소를 추가해간다. 
                    """
                    
                    for column_name in user_cols[key]:                        
                        seq_cnt += 1

                        seq_items = list()

                        for i in range(batch_size):
                            temp = data[column_name][i]
                            temp = temp[temp.nonzero().squeeze().detach()]
                            temp = temp.view(-1)
                            
                            seq_items.append(temp)

                        seq_items = torch.cat(seq_items)

                        user_seq_data.append(seq_items)

            lS_i = user_single_data + user_seq_data


            # offset 설정, 마지막 시퀀스 길이는 알필요없음
            seq_offset = list(it.accumulate(seq_offset[:-1]))
            # offset starts with zero
            seq_offset.insert(0, 0)
            # print("seq_offset", seq_offset)

            for i in range(single_cnt):
                tmp = [i for i in range(batch_size)]
                user_lS_o.append(tmp)
            

            
            for i in range(seq_cnt):
                user_lS_o.append(seq_offset)
            

        except Exception as e:
            print("user gen_sparse_factor", e)

        # product
        seq_cnt = 0
        single_cnt = 0
        prod_cols = recsys.dataset.col_name['product']['sparse']

        try:
            import itertools as it
            batch_size = cf().path["data"]["batch_size"]
            
            # offset list를 변형 가능한 꼴로 변환시켜
            #seq_offset = np.array(data["offset"].view(-1))
            
            # Padded at Head 
            seq_offset = np.array(data["offset"].view(-1))
            #

            # product
            prod_single_data = list()
            prod_seq_data = list()

            for key in prod_cols.keys():

                if key == "single":

                    for column_name in prod_cols[key]:
                        single_cnt += 1
                        #print(column_name)
                        # prod_single_data.append(data[column_name])
                        prod_single_data.append(data[column_name])

                elif key == "seq":

                    for column_name in prod_cols[key]:

                        seq_cnt += 1
                        seq_items = list()

                        for i in range(batch_size):
                            # original 
                            # seq_items.append(data[column_name][i])
                            temp = data[column_name][i]
                            temp = temp[temp.nonzero().squeeze().detach()]
                            temp = temp.view(-1)
                            
                            seq_items.append(temp)

                        seq_items = torch.cat(seq_items)
                        prod_seq_data.append(seq_items)
            

            # product ls_i
            prd_ls_i = prod_single_data + prod_seq_data

            lS_i += prd_ls_i
            

            #print(seq_offset)
            # offset 설정, 마지막 시퀀스 길이는 알필요없음
            # seq_offset = list(it.accumulate(seq_offset[:-1]))
            # Padded at Head
            seq_offset = list(it.accumulate(seq_offset[:-1]))
            # print(seq_offset)
            #

            # offset starts with zero
            seq_offset.insert(0, 0)
            
            for i in range(single_cnt):
                tmp = [i for i in range(batch_size)]
                prod_lS_o.append(tmp)

            for i in range(seq_cnt):
                prod_lS_o.append(seq_offset)
            
            
            lS_o = user_lS_o + prod_lS_o

            lS_o = torch.LongTensor(lS_o)

            return lS_i, lS_o

        except Exception as e:
            print("")
            print("prod gen_sparse_factor", e)
            print("data : ", data)
            print("prod_single_data : ", prod_single_data)
            print("prod_seq_data : ", prod_seq_data)
    def loss_fn_wrap(Z, T, use_gpu, device):
        if use_gpu:
            return loss_fn(Z, T.to(device))
        else:
            return loss_fn(Z, T)
    using_gpu = False

    # gpu 장비를 사용하는지에 따라 하드웨어 속성변경
    # if torch.cuda.is_available():
    #     # torch.cuda.manual_seed_all(args.numpy_rand_seed)
    #     # torch.backends.cudnn.deterministic = True
    #     device = torch.device("cuda", 7)
    #     using_gpu = True
    # else:

    device = torch.device("cpu")
    recsys = Recsys(device)
    # recsys.load_my_state_dict()

    writer = SummaryWriter()

    #for param_tensor in recsys.model.state_dict():
    #    print("", colored(f"{param_tensor}", "blue", attrs=["bold"]), colored(f"{recsys.model.state_dict()[param_tensor].size()}", "blue", attrs=["bold"]))

    learning_rate = cf().path["model_parameter"]["learning_rate"]

    loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(recsys.model.parameters(), lr=learning_rate)
    print(colored(f"DLRM frame generate done", 'yellow'), "\n")

    # 옵티마이저의 state_dict 출력
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

    total_iter = 0
    epochs = cf().path["data"]["epoch"]
    # epoch
    k = 0
    start_vect = time.time()

    best_model_wts = copy.deepcopy(recsys.model.state_dict())
    best_acc = 0.0

    print(colored("MODEL TRAINING START", "yellow", attrs=["underline"]), "\n")
    START_TIME = time.time()

    M = eval()

    M.add_emb(recsys, writer)

    recsys.model.train()  # 모델을 학습 모드로 설정
    
    with torch.autograd.profiler.profile(False, False) as prof:
        try:
            while k < epochs:
                total_iter =0 
                k += 1
                
                for it, (dense_data, sparse_data, label) in enumerate(recsys.dataloader):
                    
                    dense_x = gen_dense_factor(dense_data)
                    lS_i, lS_o = gen_sparse_factor(sparse_data)

                    Yhat = recsys.dlrm_wrap(dense_x, lS_o, lS_i, using_gpu, device)

                    Y = label.type(torch.FloatTensor)

                    E = loss_fn_wrap(Yhat, Y, using_gpu, device)

                    try:
                        optimizer.zero_grad()
                        # backward pass
                        E.backward()
                        # optimizer
                        optimizer.step()

                    except Exception as e:
                        print("weight update error", e)
                        sys.exit(1)

                    if(it % 50 == 0):
                        print(f"{k} epoch , iteration : {it}")
                        print(colored(f"Epoch : {k}", "blue"))
                        print("Yhat : ", Yhat)
                        print("Label : ", label)
                        print("Loss : ", E)
                        current_error = M.metrics(total_iter, E, Yhat, label, writer)

                        if best_acc < current_error:
                            best_acc = current_error
                            best_model_wts = copy.deepcopy(recsys.model.state_dict())

                    #recsys.model.load_state_dict(best_model_wts)
                    #recsys.save_model()

            print(colored(f"TRAIN RUNTIME: {(time.time() - start_vect) / 60} Min", "yellow", attrs=["underline"]), "\n")

            #distributed_inference()
            M.close(writer)
            sys.exit()

        except Exception as e:
            print("train 도중", e)


if __name__ == '__main__':

    print(colored("START TRAINING SESSION...", "yellow"), "\n")
    train()