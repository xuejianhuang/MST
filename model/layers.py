from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import copy
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean

#GCN Layer
class GraphConvLayer(Module):
    def __init__(self, in_features,hidden_features, out_features,dropout,device, bias=True):
        super(GraphConvLayer, self).__init__()
        self.conv1 = GCNConv(in_features,hidden_features,add_self_loops=False,normalize=False)
        self.conv2 = GCNConv(hidden_features+in_features, out_features, add_self_loops=False,normalize=False)
        self.linear = nn.Linear(hidden_features+out_features,in_features)
        self.dropout = dropout
        self.device = device
    def forward(self, features, adjs, values, root_idx, propagation_node_num, propagation_edge_num,batch):
        features_1 = copy.copy(features.float())
        features = self.conv1(x=features,edge_index=adjs,edge_weight=values)
        # print('layers feature after conv1:',features)
        features_2 = copy.copy(features)
        root_extend = torch.zeros(len(batch), features_1.size(1)).to(self.device)
        batch_size = max(batch)+1
        for num_batch in range(batch_size):
            index = (torch.eq(batch,num_batch))
            root_extend[index] = features_1[root_idx[num_batch]]
        # print('layers first root extend:',root_extend)
        features = torch.cat([features,root_extend],1)
        features = F.leaky_relu(features)
        #todo
        features = F.dropout(features,self.dropout,training=self.training)
        # print('layers feature after root extend:', features)
        features = self.conv2(x=features, edge_index=adjs,edge_weight=values)
        features = F.leaky_relu(features)
        root_extend = torch.zeros(len(batch),features_2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (torch.eq(batch,num_batch))
            root_extend[index] = features_2[root_idx[num_batch]]
        features = torch.cat([features,root_extend],1)
        features = F.leaky_relu(self.linear(features))

        return features

#TemporalFusion
class TemporalFusion_concat(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TemporalFusion_concat, self).__init__()
        out_features = int(in_features / 2)  # not useful now
        # out_o = out_c = int(in_features / 2)
        self.weight_o = Parameter(torch.Tensor(in_features, in_features))
        self.weight_p = Parameter(torch.Tensor(in_features, in_features))
        self.weight_k = Parameter(torch.Tensor(in_features, in_features))
        self.linear = nn.Linear(3*in_features, in_features)
        nn.init.xavier_uniform_(self.weight_o.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_p.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_k.data, gain=1.667)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, h_p, h_k,h_o):
        trans_ho = torch.mm(h_o, self.weight_o)
        trans_hp = torch.mm(h_p, self.weight_p)
        trans_hk = torch.mm(h_k, self.weight_k)
        # output = torch.tanh((torch.cat((trans_ho, trans_hc), dim=1)))  # dim=1
        output = torch.tanh(self.linear(torch.cat((trans_ho, trans_hp, trans_hk), dim=1)))
        # output_p = torch.zeros(h_p.size(0),h_p.size(1))
        # output_c = torch.zeros(h_k.)
        # output = F.leaky_relu(self.linear(output))
        if self.bias is not None:
            output = output + self.bias

        h_p = output
        h_k= output
        return h_p, h_k

class TemporalFusion_single(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TemporalFusion_single, self).__init__()
        out_features = int(in_features / 2)  # not useful now
        # out_o = out_c = int(in_features / 2)
        self.weight_o = Parameter(torch.Tensor(in_features, in_features))
        self.weight_p = Parameter(torch.Tensor(in_features, in_features))
        self.weight_k = Parameter(torch.Tensor(in_features, in_features))
        self.linear = nn.Linear(2*in_features, in_features)
        # self.linear_2 = nn.Linear(2*in_features,in_features)
        nn.init.xavier_uniform_(self.weight_o.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_p.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_k.data, gain=1.667)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, h_p,h_o):
        trans_ho = torch.mm(h_o, self.weight_o)
        trans_hp = torch.mm(h_p, self.weight_p)
        output = torch.tanh(self.linear(torch.cat((trans_ho, trans_hp), dim=1)))

        if self.bias is not None:
            output = output + self.bias
        return output

class TemporalFusion_attention(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TemporalFusion_attention, self).__init__()
        out_features = int(in_features / 2)  # not useful now
        self.weight_o = Parameter(torch.Tensor(in_features, out_features))
        self.weight_p = Parameter(torch.Tensor(in_features, out_features))
        self.weight_k = Parameter(torch.Tensor(in_features, out_features))
        self.linear = nn.Linear(in_features, in_features)
        nn.init.xavier_uniform_(self.weight_o.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_p.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_k.data, gain=1.667)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

        #多头attention
        self.num_head = 1
        self.dim_model = 128
        assert self.dim_model % self.num_head == 0
        self.dim_head = self.dim_model // self.num_head
        self.fc_Q = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V = nn.Linear(self.dim_model, self.num_head * self.dim_head)

        self.attention = Scaled_Dot_Product_Attention()
        self.fc1 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        # self.fc2 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(self.dim_model)

    def forward(self, h_p, h_k,h_o):
        temp_batch_size = h_o.size()[0]
        Q = h_o.unsqueeze(1) #200,1,128
        K_p = h_p.unsqueeze(1)
        K_k = h_k.unsqueeze(1)

        K = torch.cat([K_p,K_k],dim=1)
        Q = self.fc_Q(Q)
        K = self.fc_K(K)
        V = self.fc_V(K)
        Q = Q.view(temp_batch_size*self.num_head,-1,self.dim_head)
        K = K.view(temp_batch_size * self.num_head, -1, self.dim_head)
        V = V.view(temp_batch_size * self.num_head, -1, self.dim_head)

        scale = K.size(-1) ** -0.5
        fusion_pk = self.attention(Q,K,V,scale)
        fusion_pk = fusion_pk.view(temp_batch_size,-1,self.dim_head*self.num_head)
        fusion_pk = self.fc1(fusion_pk)
        fusion_pk = self.dropout(fusion_pk)
        fusion_pk = self.layer_norm(fusion_pk)
        fusion_pk = fusion_pk.squeeze(1)

        trans_ho = torch.mm(h_o, self.weight_o)
        trans_pk = torch.mm(fusion_pk, self.weight_p)
        output = torch.tanh(torch.cat((trans_ho,trans_pk),dim=1))
        output = F.leaky_relu(self.linear(output))

        if self.bias is not None:
            output = output + self.bias

        h_p= output
        h_k = output
        return h_p,h_k

class Mean_nBatch(Module):
    def __init__(self,device,n_feature,n_output):
        super(Mean_nBatch, self).__init__()
        self.device = device
        self.linear = nn.Linear(n_feature, n_output)

    def forward(self,x,batch):
        x = scatter_mean(x,batch,dim=0)

        return x

class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        # print('----attention------------:',attention)
        context = torch.matmul(attention, V)
        return context
