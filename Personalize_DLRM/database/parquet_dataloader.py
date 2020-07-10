# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from torch.utils import data
import pandas as pd
import sys
import numpy as np
from config.config import config as cf
import os
from collections import OrderedDict
import json


class TrainData(Dataset):
    def __init__(self):
        """
        1. path 지정
        2. 학습데이터 딕셔너리로 memory load
        3. 파티셔닝된 학습데이터 컬럼명들을 딕셔너리로 저장.
        :param file_path:
        """
        super().__init__()
        self.train_data = None
        self.len = 0

        self._data = OrderedDict()
        self._label = dict()
        # 모델 와꾸생성에 사용되는 컬럼명들
        self.col_name = OrderedDict()

        # 전체 컬럼들,,,, offset 등을 포함한다.
        self.whole_col_name = OrderedDict()

        # 상품메타정보는 임시 csv 파일로 전달받아서 현재는 이 함수 사용 X
        # self.tmp_save_meta_data()
        # tmp_save_meta_data
        # sys.exit()

        self.load_train_data_test()
        self.sparse_col_len = self.gen_sparse_col_len()

        

    def __len__(self):
        
        return self.len

    # index는 __len__의 리턴 범위내에서 랜덤하게 추출
    def __getitem__(self, row_num):

        dense_data = dict()
        sparse_data = dict()

        first_layer = ['user', 'product']
        second_layer = ['dense', 'sparse']
        third_layer = ['single', 'seq']

        """
        check whether keys are in OrderedDict or not
        """
        # gen dense factors
        for f_l in first_layer:
            if f_l in self._data:

                for s_l in second_layer:
                    if s_l in self._data[f_l]:

                        for t_l in third_layer:
                            if t_l in self._data[f_l][s_l]:

                                for idx, col_name in enumerate(self.whole_col_name[f_l][s_l][t_l]):
                                    try:

                                        # dense
                                        if s_l == second_layer[0]:
                                            dense_data[col_name] = self._data[f_l][s_l][t_l][row_num][idx]

                                        # print(data[col_name])
                                        elif s_l == second_layer[1]:
                                            sparse_data[col_name] = self._data[f_l][s_l][t_l][row_num][idx]

                                    except Exception as e:
                                        print("__getitem__", e)
        print(type(sparse_data['prd_cd_cd']))
        return dense_data, sparse_data, self._label[row_num][-1]

    def gen_sparse_col_len(self):
        """
        self._data를 참조해서 전체 컬럼명 리스트 작성
        :return: self.col_name
        """
        cnt = 0
        for root_layer_key in self.col_name.keys():
            for third_layer_key in self.col_name[root_layer_key]['sparse']:
                cnt += len(self.col_name[root_layer_key]['sparse'][third_layer_key])

        return cnt

        # print(cnt)
        # sys.exit()
        # for root_layer_key in self.col_name.keys():
        #     print("-",root_layer_key)
        #     for second_layer_key in self.col_name[root_layer_key].keys():
        #         print("---",second_layer_key)
        #         for third_layer_key in self.col_name[root_layer_key][second_layer_key]:
        #             print("-----$$",third_layer_key)
        #             print(self.col_name[root_layer_key][second_layer_key][third_layer_key])

    def get_dataloader(self):
        dataloader = data.DataLoader(self,
                                     batch_size=cf().path["data"]["batch_size"],
                                     shuffle=cf().path["data"]["shuffle"],
                                     num_workers=cf().path["data"]["num_workers"],
                                     drop_last=True)

        dataloader_test = data.DataLoader(self,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=1,
                                     drop_last=False)
        
        one = 0
        zero = 0
        for k, (_,_,label) in enumerate(dataloader_test):
            print(k, label)
            print(label.item())
            if label.item() == 1.0:
                one += 1
            else:
                zero += 1
            break
        print(one, zero)
        exit()


        
        return dataloader

    def load_keys(self):
        for index, key in enumerate(self._data.keys()):
            self._keys[index] = key


    def load_train_data_test(self):
        """
        todo 학습용 데이터 적재,
        학습에 들어갈 데이터 컬럼명 리스트 작성.
        전체 데이터 컬럼명 작성 (offset 포함된것)

        :return:
        """

        first_layer = ['user', 'product']
        second_layer = ['dense', 'sparse']
        third_layer = ['single', 'seq']

        for f_l in first_layer:
            file_member1 = OrderedDict()
            root_layer_col_name = OrderedDict()

            whole_root_layer_col_name = OrderedDict()
            for s_l in second_layer:
                file_member2 = OrderedDict()
                second_layer_col_name = OrderedDict()

                whole_second_layer_col_name = OrderedDict()
                for t_l in third_layer:

                    path = f"parquet_file/partitioned_data/train/{f_l}/{s_l}/{t_l}"
                    file_list = os.listdir(path)

                    file_list_py = [file for file in file_list if file.endswith(".dms")]
                    data = list()

                    if file_list_py:
                        for file in file_list_py:
                        # if not empty
                            with open(f'{path}/{file}', 'rb') as f:

                                data.append(pd.read_parquet(f, engine='pyarrow'))

                        train_data_df = pd.concat(data, ignore_index=True)
                        train_data_df = train_data_df.set_index("idx")
                        # if f_l == 'user' and s_l =='sparse' and t_l =='single':
                        #     print(train_data_df.tail)
                        #     sys.exit()
                        train_data_df.to_csv(f"{f_l}{s_l}{t_l}.csv", mode='w')

                        if self.len == 0:
                            try:
                                self.len = train_data_df.shape[0]

                                if self.len < 1:
                                    raise Exception('empty train data')

                            except Exception as e:
                                print(e)
                                sys.exit(1)
                    else:
                        break

                    file_member2[t_l] = train_data_df.to_numpy()

                    # 모델에 사용할 컬럼만 추려낸다.
                    bad_list = ["offset"]

                    data = np.asarray(train_data_df.columns)
                    new_list = np.asarray([x for x in data if x not in bad_list])

                    second_layer_col_name[t_l] = new_list
                    whole_second_layer_col_name[t_l] = data

                file_member1[s_l] = file_member2
                root_layer_col_name[s_l] = second_layer_col_name
                whole_root_layer_col_name[s_l] = whole_second_layer_col_name

            self._data[f_l] = file_member1
            self.col_name[f_l] = root_layer_col_name


            # 전체 컬럼명 리스트
            self.whole_col_name[f_l] = whole_root_layer_col_name


        # label
        path = "parquet_file/partitioned_data/train/label"
        file_list = os.listdir(path)

        file_list_py = [file for file in file_list if file.endswith(".dms")]

        for file in file_list_py:
            # if not empty
            if file:
                with open(f'{path}/{file}', 'rb') as f:
                    data.append(pd.read_parquet(f, engine='pyarrow'))
            label_df = pd.concat(data, ignore_index=True)
            label_df = label_df.set_index("idx")

        label_df = label_df.to_numpy()

        self._label = label_df


    def tmp_save_meta_data(self):
        prd_meta = pd.read_csv("parquet_file/partitioned_data/product_meta/product.csv")
        prd_meta_dict = dict()
        col_name_list = np.array([prd for prd in prd_meta.columns])

        for col_name in col_name_list:
            # 혹시 모르니 +1,, 인덱스 0부터 시작하는지 재확인 필요함.
            prd_meta_dict[col_name] = len(pd.unique(prd_meta[col_name]))

        json_val = json.dumps(prd_meta_dict, ensure_ascii=False, indent="\t")
        with open("config/unique_product.json", 'w') as f:
            f.write(json_val)


    def save_meta_data(self):
        """
        todo parquet_file 디렉토리에 저장된 모든 파케이 파일들을 읽어들인다.
        고객 메타, 상품메타는 json 으로 info 파일생성
        :return:
        """

        # # 상품 메타
        # prd_list = list()
        # path = "parquet_file/partitioned_data/product_meta"
        # file_list = os.listdir(path)
        # product_files = [file for file in file_list if file.endswith(".csv")]
        # print(product_files)
        # # product_files = [f for f in listdir('parquet_file/train/product_meta') if isfile(join('parquet_file/train/product_meta', f))]
        # for file in product_files:
        #     with open(f'parquet_file/train/product_meta/{file}', 'rb') as f:
        #         prd_list.append(pd.read_csv(f, engine='pyarrow'))
        #
        # # 각 파일마다 개별 인덱스를 가지고있음, ignore_index로 인덱스 통일
        # prd_meta = pd.concat(prd_list, ignore_index=True)
        # prd_meta.to_csv("prd_meta.csv", mode='w')
        #
        # #상품 각 컬럼별 유니크 값의 수 저장.
        # prd_meta_dict = dict()
        # col_name_list = np.array([prd for prd in prd_meta.columns])
        #
        # for col_name in col_name_list:
        #     # 혹시 모르니 +1
        #     prd_meta_dict[col_name] = len(pd.unique(prd_meta[col_name]))+1
        #
        # json_val = json.dumps(prd_meta_dict, ensure_ascii=False, indent="\t")
        # with open("config/unique_product.json", 'w') as f:
        #     f.write(json_val)
        #
        # file_data = OrderedDict()
        #
        # for idx in range(prd_meta.shape[0]):
        #     file_member = OrderedDict()
        #     for index, col in enumerate(prd_meta.columns):
        #         file_member[col] = str(prd_meta[col][idx])
        #     file_data[idx] = file_member
        #
        # json_val = json.dumps(file_data, ensure_ascii=False, indent="\t")
        # with open("config/item_info.json", 'w') as f:
        #     f.write(json_val)

        # 고객 메타
        client_list = list()
        path = "parquet_file/partitioned_data/inference/meta/user"
        file_list = os.listdir(path)
        client_meta = [file for file in file_list if file.endswith(".dms")]

        for file in client_meta:
            with open(f'parquet_file/partitioned_data/inference/meta/user/{file}', 'rb') as f:
                client_list.append(pd.read_parquet(f, engine='pyarrow'))
        client_meta = pd.concat(client_list, ignore_index=True)

        # 고객별 각 컬럼별 유니크 값의 수 저장.
        client_meta_dict = dict()
        client_col_name_list = np.array([prd for prd in client_meta.columns])

        for col_name in client_col_name_list:
            client_meta_dict[col_name] = len(pd.unique(client_meta[col_name]))

        json_val = json.dumps(client_meta_dict, ensure_ascii=False, indent="\t")
        with open("config/unique_client.json", 'w') as f:
            f.write(json_val)

        file_data = OrderedDict()

        for idx in range(client_meta.shape[0]):
            file_member = OrderedDict()
            for index, col in enumerate(client_meta.columns):
                file_member[col] = str(client_meta[col][idx])
            file_data[idx] = file_member

        json_val = json.dumps(file_data, ensure_ascii=False, indent="\t")
        with open("config/client_info.json", 'w') as f:
            f.write(json_val)
        sys.exit()

