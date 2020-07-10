# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from recsys.Recsys import Recsys
import operator
import time
import sys
from config.config import config as cf
import os
import torch
import torch.multiprocessing as mp
import csv
import json
import numpy as np
from collections import OrderedDict

import pandas as pd

class DI(object):
    def __init__(self):

        mp.set_start_method('spawn')
        self._top_N = cf().path["inference"]["top_N"]
        self._using_gpu = cf().path["inference"]["using_gpu"]
        self._device = torch.device(cf().path["system"]["device"])

        self._client_len = 0
        self._sku = 0
        self._user_data = OrderedDict()
        self._product_data = OrderedDict()

        self.user_col_name = OrderedDict()
        self.whole_user_col_name = OrderedDict()

        self.product_col_name = OrderedDict()
        self.whole_product_col_name = OrderedDict()

        # set process count
        self._num_processes = max(1, int(mp.cpu_count() * 0.6))
        self._num_sampler_processes = max(1, int(mp.cpu_count() * 0.2))

        self._using_gpu = False
        self._sampler_flag = mp.Manager().list()

        self.load_user_raw_data()
        self.load_product_raw_data()


    def load_inf_data(self, client_index):

        """
        user, product 에서 dense, sparse로 별도로 저장
        :param client_index:
        :return:
        """

        try:
            dense = dict()

            sparse = dict()

            result = dict()

            # user data
            for f_l in self.whole_user_col_name.keys():
                for s_l in self.whole_user_col_name[f_l]:
                    for col_order, column_name in enumerate(self.whole_user_col_name[f_l][s_l]):

                        if s_l == "seq":

                            """
                            (배치 사이즈, 시퀀스) shape의 인퍼런스 데이터 확보

                            """
                            if f_l == "dense":
                                dense[column_name] = np.full((self._sku, cf().path["data"]["SEQ_LEN"]),np.array(self._user_data[f_l][s_l][client_index][col_order]))
                            else:
                                # print(column_name)
                                # todo offset 나중에 어케관리할지 논의,
                                if column_name == 'offset':
                                    tmp = np.full((self._sku),np.array(self._user_data[f_l][s_l][client_index][col_order]))

                                    # LongTensor 아니면 Embedding_bag에서 쥐랄쥐랄
                                    tmp = torch.LongTensor(tmp)
                                    sparse[column_name] = tmp

                                else:

                                    tmp = np.full((self._sku, cf().path["data"]["SEQ_LEN"]),np.array(self._user_data[f_l][s_l][client_index][col_order]))

                                    tmp = torch.LongTensor(tmp)
                                    sparse[column_name] = tmp
                        else:
                            if f_l == "dense":
                                dense[column_name] = np.full((self._sku), np.array(self._user_data[f_l][s_l][client_index][col_order]))

                            else:
                                tmp = np.full((self._sku), np.array(self._user_data[f_l][s_l][client_index][col_order]))
                                tmp = torch.LongTensor(tmp)
                                sparse[column_name] = tmp

            # product
            for f_l in self.whole_product_col_name.keys():
                for s_l in self.whole_product_col_name[f_l]:
                    for col_order, column_name in enumerate(self.whole_product_col_name[f_l][s_l]):

                        tmp = np.transpose(self._product_data[f_l][s_l])

                        if f_l == "dense":
                            dense[column_name] = tmp[col_order]
                        else:
                            tmp = tmp[col_order]
                            tmp = torch.LongTensor(tmp)
                            sparse[column_name] = tmp #tmp[col_order]

            result['dense'] = dense
            result['sparse'] = sparse
            result['client_index'] = client_index

            return result

        except Exception as e:
            print("load rawdata", e)


    def load_user_raw_data(self):
        """
        1. data dict 생성.
            * 고객과 상품 딕셔너리는 개별로 관리한다.
            1. user dict 생성
            2. product dict 생성

        :return:
        """

        first_layer = ['dense', 'sparse']
        second_layer = ['single', 'seq']

        for f_l in first_layer:

            file_member1 = OrderedDict()
            root_layer_col_name = OrderedDict()
            whole_root_layer_col_name = OrderedDict()

            for s_l in second_layer:

                path = f"parquet_file/partitioned_data/inference/user/{f_l}/{s_l}"
                file_list = os.listdir(path)

                file_list_py = [file for file in file_list if file.endswith(".dms")]
                data = list()

                if file_list_py:

                    for file in file_list_py:
                        # if not empty
                        with open(f'{path}/{file}', 'rb') as f:
                            data.append(pd.read_parquet(f, engine='pyarrow'))

                    user_data_df = pd.concat(data, ignore_index=True)
                    user_data_df = user_data_df.set_index("incs_no_cd")

                    if self._client_len == 0:
                        try:
                            self._client_len = user_data_df.shape[0]
                            # print(train_data_df.shape[0])
                            if self._client_len < 1:
                                raise Exception('empty train data')

                        except Exception as e:
                            print(e)
                            sys.exit(1)
                else:
                    break

                file_member1[s_l] = user_data_df.to_numpy()

                # 모델에 사용할 컬럼만 추려낸다.
                bad_list = ["offset"]
                data = np.asarray(user_data_df.columns)
                new_list = np.asarray([x for x in data if x not in bad_list])

                root_layer_col_name[s_l] = new_list
                whole_root_layer_col_name[s_l] = data

            self._user_data[f_l] = file_member1

            self.user_col_name[f_l] = root_layer_col_name

            # 전체 컬럼명 리스트
            self.whole_user_col_name[f_l] = whole_root_layer_col_name

    def load_product_raw_data(self):
        """
        1. data dict 생성.
            * 고객과 상품 딕셔너리는 개별로 관리한다.
            1. user dict 생성
            2. product dict 생성

        :return:
        """

        first_layer = ['dense', 'sparse']
        second_layer = ['single', 'seq']

        for f_l in first_layer:
            # print(f_l)
            file_member1 = OrderedDict()
            root_layer_col_name = OrderedDict()
            whole_root_layer_col_name = OrderedDict()

            for s_l in second_layer:

                path = f"parquet_file/partitioned_data/inference/product/{f_l}/{s_l}"
                file_list = os.listdir(path)

                file_list_py = [file for file in file_list if file.endswith(".dms")]
                data = list()

                if file_list_py:

                    for file in file_list_py:
                        # if not empty
                        with open(f'{path}/{file}', 'rb') as f:
                            data.append(pd.read_parquet(f, engine='pyarrow'))

                    product_data_df = pd.concat(data, ignore_index=True)
                    product_data_df = product_data_df.set_index("tg_prd_cd_cd", drop = False)

                    if self._sku == 0:
                        try:
                            self._sku = product_data_df.shape[0]
                            # print(train_data_df.shape[0])
                            if self._sku < 1:
                                raise Exception('empty train data')

                        except Exception as e:
                            print(e)
                            sys.exit(1)
                else:
                    break

                file_member1[s_l] = product_data_df.to_numpy()

                # 모델에 사용할 컬럼만 추려낸다.
                bad_list = ["offset"]
                data = np.asarray(product_data_df.columns)
                new_list = np.asarray([x for x in data if x not in bad_list])

                root_layer_col_name[s_l] = new_list
                whole_root_layer_col_name[s_l] = data

            self._product_data[f_l] = file_member1

            self.product_col_name[f_l] = root_layer_col_name

            # 전체 컬럼명 리스트
            self.whole_product_col_name[f_l] = whole_root_layer_col_name


    def save(self):

        processes = []

        # 추론용 모델, 저장된 state를 load
        model = Recsys(self._device)
        model.load_my_state_dict()

        raw_data_queue = mp.Queue(maxsize=self._num_processes)
        result_queue = mp.Queue(maxsize=self._num_processes)


        for idx in range(self._num_sampler_processes):

            feeding_mp = mp.Process(target=self.sampler, args=(idx, self._num_processes,
                                                              raw_data_queue,))
            feeding_mp.daemon = True
            feeding_mp.start()
            processes.append(feeding_mp)

        for i in range(self._num_processes):
            p = mp.Process(target=self.inference, args=(model, raw_data_queue, result_queue, self._top_N,))
            p.daemon = True
            p.start()
            processes.append(p)


        save_p = mp.Process(target=self.save_to_csv, args=(result_queue, raw_data_queue, self._num_processes, ))
        save_p.daemon = True
        save_p.start()
        processes.append(save_p)

        for proc in processes:
            proc.join()

    def sampler(self, index, num_process, raw_data_queue):
        """
        todo = inference queue에 데이터 넣어주기
        client, item 데이터로부터 inference data 조합을 생성하는 함수를 내부에서 call
        single process로 샘플러를 돌릴시에 속도이슈가 발생할 수 있음.
        멀티로 돌릴때와 싱글일 때의 케이스로 1차분기,
        멀티일 경우 마지막 샘플러는 feeding end sign을 큐에 넣어준다.
        :param index: sampler's index
        :param num_process: # of inferences
        :param raw_data_queue:
        :return: None
        """

        offset = int(self._client_len / self._num_sampler_processes)
        start = index * offset

        if self._num_sampler_processes != 1:
            try:
                # 마지막 샘플러가 아닐경우
                if index < self._num_sampler_processes-1:
                    for i in range(start, start + offset):
                        raw_data_queue.put(self.load_inf_data(i), block=True)
                    # job end
                    self._sampler_flag.append([1])

                # 마지막 샘플러일 경우
                else:
                    for i in range(start, self._client_len):
                        raw_data_queue.put(self.load_inf_data(i), block=True)

                    # 마지막놈은 다른 샘플러작업이 종료되면 end 넣고 나와
                    while True:
                        if len(self._sampler_flag) == (self._num_sampler_processes - 1):
                            for idx in range(num_process):
                                raw_data_queue.put(cf().path["inference"]["feeding_end"], block=True)
                            break

            except Exception as e:
                print("sampler", e)
                sys.exit(1)

        else:
            try:
                for i in range(self._client_len):
                    data = self.load_inf_data(i)
                    raw_data_queue.put(data, block=True)

                for end in range(num_process):
                    raw_data_queue.put(cf().path["inference"]["feeding_end"], block=True)

            except Exception as e:
                print("p_sampler", e)
                sys.exit(1)

    def inference(self, model, raw_data_queue, result_queue, top_N):
        """
        "end" 태그가 들어오므로 데이터는 전부 딕셔너리 타입인것은 아님
        $$분기 잘태워$$
        :param model: 추론 모델
        :param raw_data_queue: 받아올 데이터 큐
        :param result_queue: csv로 저장할 데이터 큐
        :param top_N: 반환할 상품 수
        :return: 추천 리스트
        [incs_no ,item_index #0, item_index #1, item_index #2, ... item_index#N]
        """

        proc = os.getpid()
        start_vect = time.time()
        while True:
            try:

                data = raw_data_queue.get(block=True)

                if data == cf().path["inference"]["feeding_end"]:
                    result_queue.put(proc, block=True)
                    break

                else:
                    # get incs_no
                    client_index = data['client_index']
                    with open('config/client_info.json') as json_file:
                        json_data = json.load(json_file)
                        incs_no = json_data[str(client_index)]['incs_no']

                    dense_data = data['dense']
                    sparse_data = data['sparse']

                    dense_x = self.gen_inference_dense_factor(dense_data)

                    lS_i, lS_o = self.gen_inference_sparse_factor(sparse_data)

                    unsorted_score = model.dlrm_wrap(dense_x, lS_o, lS_i, self._using_gpu, cf().path["system"]["device"]).detach().cpu().numpy()

                    # 정렬을 위해 딕셔너리로 변환
                    ordered_item_indices = dict()
                    for idx, item in enumerate(unsorted_score):
                        ordered_item_indices[idx] = item
                    # 정렬
                    # np.sort가 훨씬빠르다.???????????????
                    sdict = sorted(ordered_item_indices.items(), key=operator.itemgetter(1), reverse=True)

                    # top N개 추출해서 딕셔너리 value로
                    items = list()
                    cnt = 0

                    items.append(incs_no)
                    for idx, item in enumerate(sdict):

                        with open('config/item_info.json') as json_file:
                            json_data = json.load(json_file)
                            prd_nm = json_data[str(sdict[idx][0])]['prd_nm']

                        items.append(prd_nm)
                        cnt += 1
                        if cnt >= top_N:
                            break

                    result_queue.put(items, block=True)

            except Exception as e:
                print("inference error ", e)

        print(f"{proc}'s serving Runtime: {(time.time() - start_vect) / 60} Minutes")

    def save_to_csv(self, result_queue, raw_data_queue, num_processes):
        cnt = 0
        proc = os.getpid()
        start_vect = time.time()
        try:
            file_name = f'serving_data/personalize_rec_{time.strftime("%m%d%H%M")}.csv'
            csvfile = open(file_name, "w",encoding='euc-kr', newline="")

            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["client_id - rec_list"])

            while True:
                rec_list = result_queue.get(block=True)

                # pid 받으면
                if isinstance(rec_list, int):
                    cnt += 1
                    if cnt == num_processes:

                        result_queue.close()
                        raw_data_queue.close()
                        result_queue.join_thread()
                        raw_data_queue.join_thread()

                        print(f"{proc}'s saving to csv runtime: {(time.time() - start_vect) / 60} Minutes")
                        break
                else:
                    csvwriter.writerow([rec_list])

        except Exception as e:
            print("save_to_csv", e)

    def gen_inference_dense_factor(self, data):

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
                    tmp = np.mean(tmp, axis=1)
                    items.append(tmp)

                else:
                    items.append(tmp)

            items = np.array(items)
            items = items.transpose()

            result = torch.Tensor(items)

            return result
        except Exception as e:
            print("gen_dense_factor", e)


    def gen_inference_sparse_factor(self, data):

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

        user_cols = self.user_col_name['sparse']

        # user
        try:
            import itertools as it

            # offset list를 변형 가능한 꼴로 변환시켜
            seq_offset = np.array(data["offset"].view(-1))

            # user
            user_single_data = list()
            user_seq_data = list()

            for key in user_cols.keys():

                if key == "single":

                    for column_name in user_cols[key]:
                        single_cnt += 1
                        user_single_data.append(data[column_name])

                elif key == "seq":
                    """
                    각 컬럼에 배치 사이즈만큼의 길이씩 원소를 추가해간다. 
                    """

                    for column_name in user_cols[key]:

                        seq_cnt += 1

                        seq_items = list()

                        for i in range(self._sku):
                            seq_items.append(data[column_name][i])

                        seq_items = torch.cat(seq_items)
                        user_seq_data.append(seq_items)

            # notice the order
            lS_i = user_single_data + user_seq_data

            # offset 설정, 마지막 시퀀스 길이는 알필요없음
            seq_offset = list(it.accumulate(seq_offset[:-1]))

            # offset starts with zero
            seq_offset.insert(0, 0)

            for i in range(single_cnt):
                tmp = [i for i in range(self._sku)]

                user_lS_o.append(tmp)

            for i in range(seq_cnt):
                user_lS_o.append(seq_offset)

        except Exception as e:
            print("user gen_sparse_factor", e)

        # product
        seq_cnt = 0
        single_cnt = 0

        prod_cols = self.product_col_name['sparse']

        try:
            import itertools as it

            # offset list를 변형 가능한 꼴로 변환시켜
            seq_offset = np.array(data["offset"].view(-1))

            # product
            prod_single_data = list()
            prod_seq_data = list()

            for key in prod_cols.keys():

                if key == "single":

                    for column_name in prod_cols[key]:
                        single_cnt += 1

                        prod_single_data.append(data[column_name])

                elif key == "seq":

                    for column_name in prod_cols[key]:

                        seq_cnt += 1
                        seq_items = list()

                        for i in range(self._sku):
                            seq_items.append(data[column_name][i])

                        seq_items = torch.cat(seq_items)

                        prod_seq_data.append(seq_items)

            # product ls_i
            prd_ls_i = prod_single_data + prod_seq_data

            lS_i += prd_ls_i

            # offset 설정, 마지막 시퀀스 길이는 알필요없음
            seq_offset = list(it.accumulate(seq_offset[:-1]))
            # offset starts with zero
            seq_offset.insert(0, 0)

            for i in range(single_cnt):
                tmp = [i for i in range(self._sku)]
                prod_lS_o.append(tmp)

            for i in range(seq_cnt):
                prod_lS_o.append(seq_offset)

            lS_o = user_lS_o + prod_lS_o

            try:
                lS_o = torch.LongTensor(lS_o)
            except Exception as e:
                print("ehrefihdf", e)

            return lS_i, lS_o

        except Exception as e:
            print("prod gen_sparse_factor", e)

def distributed_inference():
    di = DI()
    di.save()

if __name__ == '__main__':
    distributed_inference()