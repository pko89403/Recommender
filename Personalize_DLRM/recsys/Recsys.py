# -*- coding: utf-8 -*-
from recsys.dlrm import DLRM_Net
import torch
import numpy as np
from config.config import config as cf
import sys
import os
import json
#from database.parquet_dataloader import TrainData
from database.parquet_dataloader_tb_recommend_raw import TrainData
from termcolor import colored

class Recsys(object):

    def __init__(self, device):

        self.learning_rate = cf().path["model_parameter"]["learning_rate"]

        # tensorboard
        self.emb_l_colName = list()

        self.dataset = TrainData()
        self.dataloader = self.dataset.get_dataloader()


        print(colored("generating DLRM frame", "yellow"))
        self.ln_emb = self.ln_emb_generator()
        self.ln_bot = self.ln_bot_generator()
        self.ln_top = self.ln_top_generator()

        self.model = DLRM_Net(
            cf().path["model_parameter"]["m_spa"],

            self.ln_emb,
            self.ln_bot,
            self.ln_top,
            arch_interaction_op=cf().path["model_parameter"]["arch_interaction_op"],
            arch_interaction_itself=cf().path["model_parameter"]["arch_interaction_itself"],
            sigmoid_bot=cf().path["model_parameter"]["sigmoid_bot"],
            sigmoid_top=self.ln_top.size - 2,
            sync_dense_params=cf().path["model_parameter"]["sync_dense_params"],
            loss_threshold=cf().path["model_parameter"]["loss_threshold"],

            ndevices=cf().path["model_parameter"]["ndevices"],
            qr_flag=cf().path["model_parameter"]["qr_flag"],
            qr_operation=cf().path["model_parameter"]["qr_operation"],
            qr_collisions=cf().path["model_parameter"]["qr_collisions"],
            qr_threshold=cf().path["model_parameter"]["qr_threshold"],
            md_flag=cf().path["model_parameter"]["md_flag"],
            md_threshold=cf().path["model_parameter"]["md_threshold"])

        # if device != "cpu":
        #     print("not cpu")
        #     self._model = self._model.to(device)

    def dlrm_wrap(self, X, lS_o, lS_i, use_gpu, device):
        if use_gpu:
            # lS_i can be either a list of tensors or a stacked tensor.
            # Handle each case below:
            lS_i = [S_i.to(device) for S_i in lS_i] if isinstance(lS_i, list) else lS_i.to(device)
            lS_o = [S_o.to(device) for S_o in lS_o] if isinstance(lS_o, list) else lS_o.to(device)
            return self.model(X.to(device), lS_o, lS_i)

        else:
            return self.model(X, lS_o, lS_i)

    def ln_emb_generator(self):
        """
        1. unique_client, product에서 key, value 가져와서 딕셔너리에 저장.
        2. train data load해서  seq, single 순으로 key를 저장.
        3. key 순서대로 value를 저장

        :return: ls_i와 동일한 순서의 학습데이터 count list
        """

        unique_count_dict = dict()
        col_list = list()

        try:
            with open('config/unique_client_tb_recommend_raw.json') as json_file:
                json_data = json.load(json_file)
                for key in json_data:
                    unique_count_dict[key] = json_data[key]

            with open('config/unique_product_tb_recommend_raw.json') as json_file:
                json_data = json.load(json_file)
                for key in json_data:
                    unique_count_dict[key] = json_data[key]


            emb_l = list()
            # todo: get sample train data (0번지)

            for first_layer_key in self.dataset.col_name.keys():
                for third_layer_key in self.dataset.col_name[first_layer_key]["sparse"]:
                    for cols in self.dataset.col_name[first_layer_key]["sparse"][third_layer_key]:
                        emb_l.append(unique_count_dict[cols])
                        self.emb_l_colName.append(cols)
            print(self.emb_l_colName)
            print(emb_l)
            emb_l = np.array(emb_l)

            return emb_l
        except Exception as e:
            print("ln_emb_generator", e)

    def ln_top_generator(self):
        """
        top layer 생성 자동화
        :return: top layer frame
        """

        ln_top = []
        j = cf().path["model_parameter"]["ln_bot_output_layer"]
        s = self.dataset.sparse_col_len


        m = cf().path["model_parameter"]["m_spa"]
        n = int((s * m + j) / j)
        y = j + sum(n for n in range(1, n))

        #print(j,s,m,n,y)
        ln_top.append(y)
        ln_top.append(int(y / 2))
        ln_top.append(int(y / 4))
        ln_top.append(1)
        ln_top = np.array(ln_top)

        return ln_top

    def ln_bot_generator(self):

        ln_bot = []

        start = 0
        for first_layer_key in self.dataset.col_name.keys():
            for third_layer_key in self.dataset.col_name[first_layer_key]["dense"]:

                if(len(self.dataset.col_name[first_layer_key]["dense"][third_layer_key]) < 1): continue

                start += self.dataset.col_name[first_layer_key]["dense"][third_layer_key].shape[0]

        ln_bot.append(start)
        # output layer = 2
        ln_bot.append(cf().path["model_parameter"]["ln_bot_output_layer"])
        ln_bot = np.array(ln_bot)

        return ln_bot

    def load_my_state_dict(self):
        """
        todo mimicking transfer learning
        :return:
        """
        try:
            current_state = self.model.state_dict()

            if os.path.exists(cf().path["system"]["model_save_path"]):
                saved_state = torch.load(cf().path["system"]["model_save_path"])
                print("loading saved model states...")
                for name, saved_param in saved_state.items():
                    if name not in current_state:
                        continue

                    if current_state[name].shape == saved_param.shape:
                        current_state[name].copy_(saved_param)

                self.model.load_state_dict(current_state, strict=False)
                self.model.eval()

        except Exception as e:
            print("load_my_state_dict", e)

    def load_model(self):
        """
        todo inference
        :return:
        """
        try:
            if os.path.exists(cf().path["system"]["model_save_path"]):
                # 일치하는 키만 가져오도록
                self.model.load_state_dict(torch.load(cf().path["system"]["model_save_path"]), strict=False)
                self.model.eval()

        except Exception as e:
            print("load_model", e)

    def save_model(self):
        try:
            torch.save(self.model.state_dict(), cf().path["system"]["model_save_path"],
                       pickle_protocol=4)

        except Exception as e:
            print("save_model", e)
            sys.exit()
