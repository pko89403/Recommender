import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        
        support = torch.mm(input, self.weight)
        
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               
class InnerProductDecoder(Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z, sampled_nodes):
        z = F.dropout(z, self.dropout, training=self.training)
        z = z[sampled_nodes]
        adj = self.act(torch.mm(z, z.t()))
        return adj

class SourceTargetInnerProductDecoder(Module):
    def __init__(self, dropout, act=torch.sigmoid, **kwargs):
        super(SourceTargetInnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act 
        
    def forward(self, z, sampled_nodes):
        z = F.dropout(z, self.dropout, training=self.training)
        z = z[sampled_nodes]
        
        node_emb_size = z.shape[1]

        z_source = z[:, 0:int(node_emb_size/2)]
        z_target = z[:, int(node_emb_size/2):node_emb_size]
        
        adj = self.act(torch.mm(z_source, z_target.t()))
        return adj

class GravityInspiredDecoder(Module):
    def __init__(self, dropout, act=torch.sigmoid, normalization=True, pop_bias=1., **kwargs):
        super(GravityInspiredDecoder, self).__init__(**kwargs)
        self.dropout = dropout 
        self.act = act 
        self.normalization = normalization 
        self.pop_bias = pop_bias
        
    def forward(self, z, sampled_nodes):
        z = F.dropout(z, self.dropout, training=self.training)
        z = z[sampled_nodes]
        
        node_embed_size = z.shape[1]        

        z_mass = z[:, node_embed_size-1: node_embed_size]
        mass = z_mass.repeat(1, z_mass.shape[0])

        if self.normalization:
            z = F.normalize(z[:, 0:node_embed_size-1], p=2.0, dim=1)    
        else:
            z = z[:, 0:node_embed_size-1]
        
        # Gravity-Inspired Decoding        
        dist = torch.cdist(z, z)
        adj = self.act(mass - torch.mul(self.pop_bias, dist))
        return adj               