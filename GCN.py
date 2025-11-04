from torch import nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
import math

class GraphConvolution(nn.Module):
    def __init__(self, feature_numbers, out_numbers, bias=False) -> None:
        super().__init__()
        self.bias = bias
        self.w = Parameter(torch.FloatTensor(feature_numbers, out_numbers))
        if bias:
            self.b = Parameter(torch.FloatTensor(out_numbers))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        if self.bias is not False:
            self.b.data.uniform_(-stdv, stdv)
            
    def forward(self, x, adj):
        support = torch.mm(x, self.w)
        out = torch.spmm(adj, support)
        if self.bias:
            out = out + self.b

        return out


class NodeClassificationGCNN(nn.Module):

    def __init__(self, feature_num, node_representation_dim, nclass, droupout=0.2, bias=False) -> None:
        super().__init__()
        self.gconv1 = GraphConvolution(feature_num, node_representation_dim, bias)
        self.gconv2 = GraphConvolution(node_representation_dim, nclass, bias)
        self.dropout = droupout

    def forward(self, x, adj):
        x = F.relu(self.gconv1(x, adj))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.gconv2(x, adj))
        return F.log_softmax(x, dim=1)
