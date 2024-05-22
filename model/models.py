from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from layers import *
from transformers import BertModel, BertConfig

# Dual Dynamic Graph (Dynamic Propagation Graph + Dynamic User Interaction Graph)
class dual_DynamicGCN(nn.Module):
    def __init__(self,text_embedding,idx2user_dict,mid2bert_tokenizer, bert_path,config,user_features=10):
        super(dual_DynamicGCN, self).__init__()
        self.text_embedding=text_embedding
        self.mid2bert_tokenizer = mid2bert_tokenizer
        self.idx2user_dict = idx2user_dict
        self.config = config
        self.n_output = config.n_class
        self.device = config.device
        self.n_hidden = config.n_hidden
        self.dropout = config.dropout
        self.n_feature = self.config.hidden_dim
        self.embedding_dim=text_embedding.shape[1]
        self.user_features = user_features

        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding.from_pretrained(self.text_embedding,freeze=self.config.embedding_freeze)
        self.fc_embedding = nn.Linear(self.embedding_dim, self.n_feature )
        # ------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        bert_outputdim = self.bert.config.hidden_size
        self.bert_fc = nn.Linear(bert_outputdim, self.n_feature)

        self.fc_user=nn.Linear(self.user_features,self.n_feature)


        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.bn_stack_propagation = nn.ModuleList()
        self.layer_stack_user = nn.ModuleList()
        self.bn_stack_user = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(self.n_hidden):
            self.layer_stack_propagation.append(GraphConvLayer(self.n_feature, self.n_feature, self.n_feature, self.dropout, self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_user.append(GraphConvLayer(self.n_feature, self.n_feature, self.n_feature, self.dropout, self.device))
            self.bn_stack_user.append(nn.BatchNorm1d(self.n_feature))
            if self.config.tf=='attention':
                self.temporal_cells.append(TemporalFusion_attention(self.n_feature, self.n_feature))
            elif self.config.tf=='concat':
                self.temporal_cells.append(TemporalFusion_concat(self.n_feature, self.n_feature))


        self.mean = Mean_nBatch(self.device, self.n_feature, self.n_feature)
        self.fc_rumor_1 = nn.Linear(self.n_feature * 2, self.n_feature)
        self.fc_rumor_2 = nn.Linear(self.n_feature, self.n_output)
    def forward(self, data):
        x_propagation = []
        x_user=[]
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        root_idx_p = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            output = self.bert(input_ids, attention_mask=attention_mask_bert)
            text_encoding = output['pooler_output']
            text_encoding = self.bert_fc(text_encoding)
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert, text_encoding], dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)

        for idx in data.x:
            idx = idx.item()
            text_embedding = self.embedding(torch.tensor([idx]).to(self.device))
            x_propagation.append(text_embedding.detach().cpu().numpy())
            user_features = self.idx2user_dict[idx]
            x_user.append(user_features)

        x_propagation = torch.tensor(x_propagation).to(self.device)
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_propagation=self.fc_embedding(x_propagation)
        x_user = torch.tensor(x_user,dtype=torch.float).to(self.device)
        x_user=self.fc_user(x_user)  #10->128

        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        last_x = x_propagation
        for i, gcn_layer in enumerate(self.layer_stack_propagation):
            x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch)
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_propagation = F.leaky_relu(x_propagation)
            x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            x_user = self.bn_stack_user[i](
                self.layer_stack_user[i](x_user, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch))
            x_user = F.leaky_relu(x_user)
            x_user = F.dropout(x_user, self.dropout, training=self.training)
            x_propagation,x_user = self.temporal_cells[i](x_propagation,x_user,last_x)  # temporal encoding
            last_x = x_propagation
        x = self.mean(x_propagation, data.batch)
        x = x.squeeze(1)
        x_fusion = torch.cat([x,text_encoding_bert],dim=1)
        x_fusion = F.leaky_relu(self.fc_rumor_1(x_fusion))
        x_fusion = self.fc_rumor_2(x_fusion)
        x_fusion = torch.sigmoid(x_fusion)
        return x_fusion

# Only dynamic propagation graph, initializing the nodes in the graph randomly using embeddings
class propagation_DynamicGCN(nn.Module):
    def __init__(self, n_nodes, mid2bert_tokenizer, bert_path,config):
        super(propagation_DynamicGCN, self).__init__()
        self.n_nodes = n_nodes
        self.mid2bert_tokenizer = mid2bert_tokenizer
        self.config = config
        self.n_output = config.n_class
        self.device = config.device
        self.n_hidden = config.n_hidden
        self.dropout = config.dropout
        self.n_feature = self.config.hidden_dim
        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding(self.n_nodes, config.embedding_dim)

        #------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, self.n_feature)

        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.bn_stack_propagation = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(self.n_hidden):
            self.layer_stack_propagation.append(GraphConvLayer(self.n_feature, self.n_feature, self.n_feature, self.dropout, self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalFusion_single(self.n_feature, self.n_feature))

        self.mean = Mean_nBatch(self.device, self.n_feature, self.n_feature)
        self.fc_rumor_1 = nn.Linear(self.n_feature * 2, self.n_feature)
        self.fc_rumor_2 = nn.Linear(self.n_feature, self.n_output)

    def forward(self, data):
        x_propagation = []
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        root_idx_p = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            output = self.bert(input_ids,attention_mask=attention_mask_bert)
            text_encoding=output['pooler_output']
            text_encoding = self.fc(text_encoding)
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)

        for idx in data.x:
            idx = idx.item()
            text_embedding = self.embedding(torch.LongTensor([idx]).to(self.device))
            x_propagation.append(text_embedding.detach().cpu().numpy())
        x_propagation = torch.tensor(x_propagation)
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_propagation = x_propagation.to(self.device)
        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        last_x = x_propagation
        for i, gcn_layer in enumerate(self.layer_stack_propagation):
            x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch)
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_propagation = F.leaky_relu(x_propagation)
            x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            x_propagation = self.temporal_cells[i](x_propagation,last_x)  # temporal encoding
            last_x=x_propagation
        x = self.mean(x_propagation, data.batch)
        x = x.squeeze(1)
        x_fusion = torch.cat([x,text_encoding_bert],dim=1)
        x_fusion = F.leaky_relu(self.fc_rumor_1(x_fusion))
        x_fusion = self.fc_rumor_2(x_fusion)
        x_fusion = torch.sigmoid(x_fusion)
        return x_fusion

# Only dynamic propagation graph, initializing the nodes in the graph using textual semantic features
class semantic_DynamicGCN(nn.Module):
    def __init__(self,text_embedding,mid2bert_tokenizer, bert_path,config):
        super(semantic_DynamicGCN, self).__init__()
        self.text_embedding=text_embedding
        self.mid2bert_tokenizer = mid2bert_tokenizer
        self.config = config
        self.n_output = config.n_class
        self.device = config.device
        self.n_hidden = config.n_hidden
        self.dropout = config.dropout
        self.n_feature = self.config.hidden_dim
        self.embedding_dim=text_embedding.shape[1]
        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding.from_pretrained(self.text_embedding,freeze=self.config.embedding_freeze)
        self.embedding_fc = nn.Linear(self.embedding_dim, self.n_feature )

        # ------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        bert_outputdim = self.bert.config.hidden_size
        self.bert_fc = nn.Linear(bert_outputdim, self.n_feature)

        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.bn_stack_propagation = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(self.n_hidden):
            self.layer_stack_propagation.append(GraphConvLayer(self.n_feature, self.n_feature, self.n_feature, self.dropout, self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalFusion_single(self.n_feature, self.n_feature))
        self.mean = Mean_nBatch(self.device, self.n_feature, self.n_feature)
        self.fc_rumor_1 = nn.Linear(self.n_feature*2,self.n_feature)
        self.fc_rumor_2 = nn.Linear(self.n_feature,self.n_output)
    def forward(self, data):
        x_propagation = []
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        root_idx_p = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            output = self.bert(input_ids, attention_mask=attention_mask_bert)
            text_encoding = output['pooler_output']
            text_encoding = self.bert_fc(text_encoding)
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)

        for idx in data.x:
            idx = idx.item()
            text_embedding = self.embedding(torch.tensor([idx]).to(self.device))
            x_propagation.append(text_embedding.detach().cpu().numpy())

        x_propagation = torch.tensor(x_propagation).to(self.device)
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_propagation = self.embedding_fc(x_propagation)
        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        last_x = x_propagation
        for i, gcn_layer in enumerate(self.layer_stack_propagation):
            x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch)
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_propagation = F.leaky_relu(x_propagation)
            x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            x_propagation = self.temporal_cells[i](x_propagation,last_x)  # temporal encoding
            last_x = x_propagation
        x = self.mean(x_propagation, data.batch)
        x = x.squeeze(1)
        x_fusion = torch.cat([x,text_encoding_bert],dim=1)
        x_fusion = F.leaky_relu(self.fc_rumor_1(x_fusion))
        x_fusion = self.fc_rumor_2(x_fusion)
        x_fusion = torch.sigmoid(x_fusion)
        return x_fusion

#Only dynamic user interaction graph, initializing the nodes in the graph using user social features.
class user_DynamicGCN(nn.Module):
    def __init__(self, idx2user_dict, mid2bert_tokenizer, bert_path,config,user_features=10):
        super(user_DynamicGCN, self).__init__()
        self.idx2user_dict = idx2user_dict
        self.mid2bert_tokenizer = mid2bert_tokenizer
        self.config = config
        self.n_output = config.n_class
        self.device = config.device
        self.n_hidden=config.n_hidden
        self.dropout = config.dropout
        self.n_feature = self.config.hidden_dim
        self.user_features = user_features

        #------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        bert_outputdim = self.bert.config.hidden_size
        self.bert_fc = nn.Linear(bert_outputdim, self.n_feature)

        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.bn_stack_propagation = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(self.n_hidden):
            self.layer_stack_propagation.append(GraphConvLayer(self.n_feature, self.n_feature, self.n_feature, self.dropout, self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalFusion_single(self.n_feature, self.n_feature))
        self.mean = Mean_nBatch(self.device, self.n_feature, self.n_feature)
        self.fc_rumor_1 = nn.Linear(self.n_feature*2,self.n_feature)
        self.fc_rumor_2 = nn.Linear(self.n_feature,self.n_output)
        self.fc_layer = nn.Linear(self.user_features, self.n_feature)

    def forward(self, data):
        x_propagation = []
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        root_idx_p = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            output = self.bert(input_ids, attention_mask=attention_mask_bert)
            text_encoding = output['pooler_output']
            text_encoding = self.bert_fc(text_encoding)
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert, text_encoding], dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)

        for idx in data.x:
            idx = idx.item()
            user_features=self.idx2user_dict[idx]
            x_propagation.append(user_features)
        x_propagation = torch.tensor(x_propagation,dtype=torch.float).to(self.device)
        x_propagation=self.fc_layer(x_propagation)  #10->128
        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progataion_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        last_x = x_propagation
        for i, gcn_layer in enumerate(self.layer_stack_propagation):
            x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progataion_edge_num[i],data.batch)
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_propagation = F.leaky_relu(x_propagation)
            x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            x_propagation = self.temporal_cells[i](x_propagation,last_x)  # temporal encoding
            last_x = x_propagation
        x = self.mean(x_propagation, data.batch)
        x = x.squeeze(1)
        x_fusion = torch.cat([x,text_encoding_bert],dim=1)
        x_fusion = F.leaky_relu(self.fc_rumor_1(x_fusion))
        x_fusion = self.fc_rumor_2(x_fusion)
        x_fusion = torch.sigmoid(x_fusion)
        return x_fusion

#Remove temporal fusion unit
class dual_DynamicGCN_wotf(nn.Module):
    def __init__(self,text_embedding,idx2user_dict,mid2bert_tokenizer, bert_path,config,user_features=10):
        super(dual_DynamicGCN_wotf, self).__init__()
        self.text_embedding=text_embedding
        self.mid2bert_tokenizer = mid2bert_tokenizer
        self.idx2user_dict = idx2user_dict
        self.config = config
        self.n_output = config.n_class
        self.device = config.device
        self.n_hidden = config.n_hidden
        self.dropout = config.dropout
        self.n_feature = self.config.hidden_dim
        self.embedding_dim=text_embedding.shape[1]
        self.user_features = user_features

        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding.from_pretrained(self.text_embedding,freeze=self.config.embedding_freeze)
        self.fc_embedding = nn.Linear(self.embedding_dim, self.n_feature )

        # ------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        bert_outputdim = self.bert.config.hidden_size
        self.bert_fc = nn.Linear(bert_outputdim, self.n_feature)

        self.fc_user=nn.Linear(self.user_features,self.n_feature)


        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.bn_stack_propagation = nn.ModuleList()
        self.layer_stack_user = nn.ModuleList()  # TODO class initiate
        self.bn_stack_user = nn.ModuleList()
        for i in range(self.n_hidden):
            self.layer_stack_propagation.append(GraphConvLayer(self.n_feature, self.n_feature, self.n_feature, self.dropout, self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_user.append(GraphConvLayer(self.n_feature, self.n_feature, self.n_feature, self.dropout, self.device))
            self.bn_stack_user.append(nn.BatchNorm1d(self.n_feature))

        self.mean = Mean_nBatch(self.device, self.n_feature, self.n_feature)
        self.fc_rumor_1 = nn.Linear(self.n_feature*3,self.n_feature)
        self.fc_rumor_2 = nn.Linear(self.n_feature,self.n_output)
    def forward(self, data):
        x_propagation = []
        x_user=[]
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        root_idx_p = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            output = self.bert(input_ids, attention_mask=attention_mask_bert)
            text_encoding = output['pooler_output']
            text_encoding = self.bert_fc(text_encoding)
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)

        for idx in data.x:
            idx = idx.item()
            text_embedding = self.embedding(torch.tensor([idx]).to(self.device))
            x_propagation.append(text_embedding.detach().cpu().numpy())
            user_features = self.idx2user_dict[idx]
            x_user.append(user_features)

        x_propagation = torch.tensor(x_propagation).to(self.device)
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_propagation=self.fc_embedding(x_propagation)
        x_user = torch.tensor(x_user,dtype=torch.float).to(self.device)
        x_user=self.fc_user(x_user)  #10->128

        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        #last_x = x_propagation
        for i, gcn_layer in enumerate(self.layer_stack_propagation):
            x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch)
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_propagation = F.leaky_relu(x_propagation)
            x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            x_user = self.bn_stack_user[i](
                self.layer_stack_user[i](x_user, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch))
            x_user = F.leaky_relu(x_user)
            x_user = F.dropout(x_user, self.dropout, training=self.training)
            #x_propagation,x_user = self.temporal_cells[i](x_propagation,x_user,last_x)  # temporal encoding
            #last_x = x_propagation
        x_p = self.mean(x_propagation, data.batch)
        x_p = x_p.squeeze(1)
        x_u = self.mean(x_user, data.batch)
        x_u = x_u.squeeze(1)

        x_fusion = torch.cat([x_p,x_u,text_encoding_bert],dim=1)
        x_fusion = F.leaky_relu(self.fc_rumor_1(x_fusion))
        x_fusion = self.fc_rumor_2(x_fusion)
        x_fusion = torch.sigmoid(x_fusion)
        return x_fusion

#Double static GCN model
class dual_StaticGCN(nn.Module):
    def __init__(self,text_embedding,idx2user_dict,mid2bert_tokenizer, bert_path,config,user_features=10):
        super(dual_StaticGCN, self).__init__()
        self.text_embedding=text_embedding
        self.mid2bert_tokenizer = mid2bert_tokenizer
        self.idx2user_dict = idx2user_dict
        self.config = config
        self.n_output = config.n_class
        self.device = config.device
        self.n_hidden = config.n_hidden
        self.dropout = config.dropout
        self.n_feature = self.config.hidden_dim
        self.embedding_dim=text_embedding.shape[1]
        self.user_features = user_features

        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding.from_pretrained(self.text_embedding,freeze=self.config.embedding_freeze)
        self.fc_embedding = nn.Linear(self.embedding_dim, self.n_feature )
        # ------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        bert_outputdim = self.bert.config.hidden_size
        self.bert_fc = nn.Linear(bert_outputdim, self.n_feature)

        self.fc_user=nn.Linear(self.user_features,self.n_feature)


        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.bn_stack_propagation = nn.ModuleList()
        self.layer_stack_user = nn.ModuleList()  # TODO class initiate
        self.bn_stack_user = nn.ModuleList()
        for i in range(self.n_hidden):
            self.layer_stack_propagation.append(GraphConvLayer(self.n_feature, self.n_feature, self.n_feature, self.dropout, self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_user.append(GraphConvLayer(self.n_feature, self.n_feature, self.n_feature, self.dropout, self.device))
            self.bn_stack_user.append(nn.BatchNorm1d(self.n_feature))

        self.mean = Mean_nBatch(self.device, self.n_feature, self.n_feature)
        self.fc_rumor_1 = nn.Linear(self.n_feature*3,self.n_feature)
        self.fc_rumor_2 = nn.Linear(self.n_feature,self.n_output)
    def forward(self, data):
        x_propagation = []
        x_user=[]
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        root_idx_p = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            output = self.bert(input_ids, attention_mask=attention_mask_bert)
            text_encoding = output['pooler_output']
            text_encoding = self.bert_fc(text_encoding)
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)

        for idx in data.x:
            idx = idx.item()
            text_embedding = self.embedding(torch.tensor([idx]).to(self.device))
            x_propagation.append(text_embedding.detach().cpu().numpy())
            user_features = self.idx2user_dict[idx]
            x_user.append(user_features)

        x_propagation = torch.tensor(x_propagation).to(self.device)
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_propagation=self.fc_embedding(x_propagation)
        x_user = torch.tensor(x_user,dtype=torch.float).to(self.device)
        x_user=self.fc_user(x_user)  #10->128

        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        #last_x = x_propagation
        for i, gcn_layer in enumerate(self.layer_stack_propagation):
            x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[2], x_propagation_node_values[2], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[2],data.batch)
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_propagation = F.leaky_relu(x_propagation)
            x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            x_user = self.bn_stack_user[i](
                self.layer_stack_user[i](x_user, x_propagation_node_indices[2], x_propagation_node_values[2], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[2],data.batch))
            x_user = F.leaky_relu(x_user)
            x_user = F.dropout(x_user, self.dropout, training=self.training)
            #x_propagation,x_user = self.temporal_cells[i](x_propagation,x_user,last_x)  # temporal encoding
            #last_x = x_propagation
        x_p = self.mean(x_propagation, data.batch)
        x_p = x_p.squeeze(1)
        x_u = self.mean(x_user, data.batch)
        x_u = x_u.squeeze(1)
        x_fusion = torch.cat([x_p,x_u,text_encoding_bert],dim=1)
        x_fusion = F.leaky_relu(self.fc_rumor_1(x_fusion))
        x_fusion = self.fc_rumor_2(x_fusion)
        x_fusion = torch.sigmoid(x_fusion)
        return x_fusion