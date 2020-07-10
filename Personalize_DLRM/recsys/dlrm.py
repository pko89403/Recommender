# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch
import torch.nn as nn
import sys
from config.config import config as cf

class DLRM_Net(nn.Module):

    def __init__(self, m_spa=None, ln_emb=None, ln_bot=None, ln_top=None, arch_interaction_op=None,
                 arch_interaction_itself=False, sigmoid_bot=-1, sigmoid_top=-1, sync_dense_params=True,
                 loss_threshold=0.0,
                 ndevices=-1, qr_flag=False, qr_operation="mult", qr_collisions=0, qr_threshold=200, md_flag=False,
                 md_threshold=200, ):
        super(DLRM_Net, self).__init__()

        if (
                (m_spa is not None)
                and (ln_emb is not None)
                and (ln_bot is not None)
                and (ln_top is not None)
                and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold

            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold
            # create operators
            if ndevices <= 1:
                self.emb_l = self.create_emb(m_spa, ln_emb)

            
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)

            self.top_l = self.create_mlp(ln_top, sigmoid_top)

    """
    moduleList를 생성, 레이어를 append 해준 후 리스트를 반환.
    sigmoid_layer에는 히든 레이어의 수를 넘겨줌.
    """

    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()

        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]
            # construct fully connected operator

            LL = nn.Linear(int(n), int(m), bias=True)
            #nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')

            nn.init.kaiming_uniform_(LL.weight)
            #nn.init.kaiming_uniform_(LL.bias.data)

            layers.append(LL)
            
            
            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                break
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
        print(layers)
        return torch.nn.Sequential(*layers)

    """
    Modulelist에다가 생성한 임베딩을 붙여준다. 
    각각의 임베딩을 Embedding bag이라고하는데 이건 뭔지모르겠음. 이거 학습가능하도록 웨이트를 초기화시켜주는데, 이것도 아직 왜 이러는지 모르겠음.
    임베딩을 학습시키나보네
    """

    def create_emb(self, m, ln):
        emb_l = nn.ModuleList()
        for i in range(0, ln.size):
            n = ln[i]
            
            EE = nn.EmbeddingBag(n+1, m, mode="mean", sparse=False)
            init_range =(2.0 / (n + m)) ** 0.5 # Xaiver init
            EE.weight.data.uniform_(-init_range, init_range)
            emb_l.append(EE)

        return emb_l

    def apply_mlp(self, x, layers):
        try:
            return layers(x)

        except Exception as e:
            print("layers : ", layers)
            print("x.shape : ", x.shape)
            print("x.value : ", x)
            print("apply_mlp", e)

    def apply_emb(self, lS_o, lS_i, emb_l):

        try:
            ly = []
            # print("------------------------")
            for k, sparse_index_group_batch in enumerate(lS_i):
                
                sparse_offset_group_batch = lS_o[k]
                test = sparse_index_group_batch
                # print(emb_l, k, emb_l[k], lS_i)
                Embedding = emb_l[k]
                # print(Embedding)
                #print(sparse_index_group_batch)
                #print(test)
                # print(k)
                # print("")
                V = Embedding(test, sparse_offset_group_batch)

                ly.append(V)

            return ly

        except Exception as e:
            #print("Embedding", emb_l[k])
            #print("sparse_index_group_batch : ", sparse_index_group_batch)
            #print("sparse_offset_group_batch : ", sparse_offset_group_batch)
            #print("k",k, "\n","Embedding", Embedding,"\n", "sparse_index_group_batch", sparse_index_group_batch, "\n",
            #      "sparse_offset_group_batch", sparse_offset_group_batch,"\n", "V", V)
            print(
                "apply_embff", e)
            exit()

    def interact_features(self, x, ly):
        try:
            if self.arch_interaction_op == "dot":
                # concatenate dense and sparse features
                (batch_size, d) = x.shape

                T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))

                #print("T.shape", T.shape)
                # perform a dot product
                Z = torch.bmm(T, torch.transpose(T, 1, 2))

                
                _, ni, nj = Z.shape
                offset = 1 if self.arch_interaction_itself else 0
                li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
                lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])

                Zflat = Z[:, li, lj]
                # concatenate dense features and interactions
                R = torch.cat([x] + [Zflat], dim=1)
                #print("R.shape", R.shape)
            return R
        except Exception as e:
            print("interact_features", e)

    # override
    def forward(self, dense_x, lS_o, lS_i):
        # Todo ndevice는 뭐야??? gpu 장비 대수를 말하나보다.. 장비가 여러개면 갯수만큼 사용할 수 있음(parallen_forward). cpu일경우에는(sequential) -1
        if self.ndevices <= 1:
            return self.sequential_forward(dense_x, lS_o, lS_i)

    def sequential_forward(self, dense_x, lS_o, lS_i):
        try:
            x = self.apply_mlp(dense_x, self.bot_l)
            
            ly = self.apply_emb(lS_o, lS_i, self.emb_l)

            z = self.interact_features(x, ly)

            p = self.apply_mlp(z, self.top_l)
            
            # clamp output if needed
            #if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            #    z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
            #else:
            #    z = p
            z = p
            z = z.view([-1])
            return z
        except Exception as e:
            print("sequential_forward", e)
            print("x.shape : ", x.shape)
            print("ly list len : ", len(ly))
            print("z.shape : ", z.shape)
            sys.exit(1)

