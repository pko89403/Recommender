import os 
import json 

import scipy.sparse as sp 
import numpy as np
import torch 
from torch import optim
from tqdm import trange

from model.optimizer import loss_function, EarlyStopping
from model.gravity import GCNModelAE, GCNModelVAE
from util.conf_util import init_config
from util.preprocess_utils import (
    load_data,
    preprocess_graph,
    get_distribution,
    mask_test_edges_org,
    sparse_mx_to_torch_sparse_tensor
)
from util.sampling_utils import parallel_node_sampling
from util.metric_utils import get_gravity_roc_score_split


device = None

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class Trainer():
    def __init__(self, config):
        self.origin_adj = None 
        self.feat_dim = None
        self.train_adj = None
        self.adj_norm = None
        self.subgraph_num = None 
        self.subgraph_node_dist = None 
        self.subgraph_sampled = None
        self.subgraph_label = None 
        self.subgraph_norm = None 
        self.subgraph_weight = None 
        self.valid_dataset = None 
        self.test_dataset = None 
        self.model = None
        self.node_emb = None 
        
        self.dataset_path = config["dataset"]["path"]
        self.dataset_name = config["dataset"]["name"]
        self.model_name=config["model"]["name"]
        self.epoch=config["model"]["epoch"]
        self.learning_rate=config["model"]["lr"]
        self.hidden_dim1=config["model"]["layer"]["hidden_dim_1"]
        self.hidden_dim2=config["model"]["layer"]["hidden_dim_2"]
        self.dropout_rate=config["model"]["layer"]["dropout_rate"]
        self.normalize=config["model"]["normalize"]
        self.pop_bias=config["model"]["pop_bias"]
        self.earlystop_start=config["model"]["early_stop"]["start"]
        self.earlystop_patience=config["model"]["early_stop"]["patience"]
        self.earlystop_delta=config["model"]["early_stop"]["delta"]
        self.subgraph_replace=config["model"]["subgraph"]["replace"]
        self.subgraph_rate=config["model"]["subgraph"]["rate"]
        self.subgraph_method=config["model"]["subgraph"]["method"]
        self.subgraph_alpha=config["model"]["subgraph"]["alpha"]
        self.validate_rate=config["model"]["validation"]["rate"]
        self.validate_step=config["model"]["validation"]["step"]
        self.test_rate=config["model"]["test"]["rate"]
        self.test_split=config["model"]["test"]["split"]

        self.preprocess()
        self.model = self.init_model(name=self.model_name)
        
        print(self.model)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.earlystop = EarlyStopping(
            patience=self.earlystop_patience,
            verbose=True,
            delta=self.earlystop_delta,
            ckpt_path=os.path.join("artifact", "checkpoint.pt"),
            trace_func=print
        )
        self.loss_func = loss_function
        
        self.train_dataset = parallel_node_sampling(
            total_epochs=self.epoch,
            n_cores=5,
            adj=self.train_adj,
            dist=self.subgraph_node_dist,
            num_samples=self.subgraph_num,
            replace=self.subgraph_replace,
        )
                
    def init_model(self, name: str = "AE"):
        model = None

        if name == "AE":
            model = GCNModelAE(
                input_feat_dim=self.feat_dim,
                hidden_dim1=self.hidden_dim1,
                hidden_dim2=self.hidden_dim2,
                dropout=self.dropout_rate,
                normalization=self.normalize,
                pop_bias=self.pop_bias
            )

        else:
            model = GCNModelVAE(
                input_feat_dim=self.feat_dim,
                hidden_dim1=self.hidden_dim1,
                hidden_dim2=self.hidden_dim2,
                dropout=self.dropout_rate,
                normalization=self.normalize,
                pop_bias=self.pop_bias
            )        

        return model.to(device)

    def train_one_epoch(self, train_data):
        self.optimizer.zero_grad()

        adj_sampled_sparse = self.train_adj[train_data,:][:,train_data]
        adj_label = sparse_mx_to_torch_sparse_tensor(
            adj_sampled_sparse + sp.eye(adj_sampled_sparse.shape[0])
        )

        num_sampled = adj_sampled_sparse.shape[0]
        sum_sampled = adj_sampled_sparse.sum()
        
        pos_weight = float(num_sampled * num_sampled - sum_sampled) / sum_sampled
        norm = num_sampled * sum_sampled / float((num_sampled * num_sampled - sum_sampled) * 2)
        
        z, mu, logvar = self.model(
            self.features.to(device),
            self.adj_norm.to(device)
        )

        self.node_emb = z
        preds = self.model.dc(z, train_data)

        loss = self.loss_func(
            preds=preds,
            labels=adj_label.to_dense().to(device),
            mu=mu,
            logvar=logvar,
            n_nodes=num_sampled,
            norm=norm,
            pos_weight=pos_weight
        )

        loss.backward()
        self.optimizer.step()

        return loss

    def save_tensor_as_numpy(self, tensor: torch.tensor, path: str, name: str):
        file_path = f"{path}/{name}"
        np.save(file_path, tensor.cpu().detach().numpy(), allow_pickle=True)

    def preprocess(self):
        adj_orig, features, index2node = load_data(path=self.dataset_path, dataset=self.dataset_name)
        with open(os.path.join("artifact","index2node.json"), "w") as f:
            json.dump(index2node, f)

        n_nodes, feat_dim = features.shape
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape) # [np.newaxis, :] : 차원 추가
        adj_orig.eliminate_zeros()

        # adj, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_org(
        #     adj=adj_orig,
        #     test_percent=self.test_rate,
        #     val_percent=self.validate_rate
        # )

        adj = adj_orig

        node_distribution = get_distribution(
            measure=self.subgraph_method,
            alpha=self.subgraph_alpha,
            adj=adj
        )
        num_samples = int(adj.shape[0] * self.subgraph_rate)
        print(f"num of total nodes : {adj.shape[0]}")
        print(f"num of sampling nodes : {num_samples}")

        adj_norm = preprocess_graph(adj_orig)

        self.feat_dim = feat_dim
        self.features = features
        self.origin_adj = adj_orig
        self.train_adj = adj_orig
        self.adj_norm = adj_norm
        self.subgraph_num = num_samples
        self.subgraph_node_dist = node_distribution
        # self.valid_dataset = [val_edges, val_edges_false]
        # self.test_dataset = [test_edges, test_edges_false]

    def train(self):
        with trange(self.epoch, unit="epoch") as pbar:
            for epoch in pbar:
                loss = self.train_one_epoch(self.train_dataset[epoch])
                pbar.set_postfix({"train_loss":loss.item()})
        
    def validation(self):
        roc_score, ap_score = get_gravity_roc_score_split(
            emb=self.node_emb,
            adj_orig=self.origin_adj,
            edges_pos=self.valid_dataset[0],
            edges_neg=self.valid_dataset[1],
            device=device,
            pop_bias=self.pop_bias,
            normalization=self.normalize,
            split_size=self.test_split
        )

        return roc_score, ap_score

    def test(self):
        roc_score, ap_score = get_gravity_roc_score_split(
            emb=self.node_emb,
            adj_orig=self.origin_adj,
            edges_pos=self.test_dataset[0],
            edges_neg=self.test_dataset[1],
            device=device,
            pop_bias=self.pop_bias,
            normalization=self.normalize,
            split_size=self.test_split
        )

        return roc_score, ap_score