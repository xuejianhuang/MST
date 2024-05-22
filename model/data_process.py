
import torch
import re
from path_zh import *
import numpy as np
from data import *
import jieba
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transformers import BertTokenizer

import warnings
warnings.filterwarnings('ignore')

def remove_single_nodepost(p_idx, p_node, y, p_root_idx):
    p_idx_new = []
    p_node_new = []
    y_new = []
    p_root_idx_new = []
    for i in range(len(p_idx)):
        if len(p_idx[i]) > 1:
            p_idx_new.append(p_idx[i])
            p_node_new.append(p_node[i])
            y_new.append(y[i])
            p_root_idx_new.append(p_root_idx[i])
    p_idx_new = np.array(p_idx_new)
    p_node_new = np.array(p_node_new)
    y_new = np.array(y_new)
    p_root_idx_new = np.array(p_root_idx_new)
    return p_idx_new,p_node_new,y_new, p_root_idx_new

def loadData(x_propagation_idx, x_propagation_node_indices, x_propagation_node_values, target, root_idx):
    data_set = GraphDataset(x_propagation_idx, x_propagation_node_indices, x_propagation_node_values, \
                            target, root_idx)
    return data_set

class StringProcess(object):
    def __init__(self):
        self.other_char = re.compile(r"[^A-Za-z0-9(),!?\'\`]", flags=0)
        self.num = re.compile(r"[+-]?\d+\.?\d*", flags=0)
        # self.url = re.compile(r"[a-z]*[:.]+\S+|\n|\s+", flags=0)
        self.url = re.compile(
                r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", flags=0)
        self.stop_words = None
        self.nlp = None

    def clean_str(self, string):
        r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
        string = string.split('http')[0]
        cleanr = re.compile('<.*?>')
        string = re.sub(cleanr,' ',string)
        string = re.sub(r4,' ',string)
        string = string.strip().lower()
        string = self.remove_stopword(string)

        return string

    def clean_str_zh(self, string):
        r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
        cleanr = re.compile('<.*?>')
        string = re.sub(cleanr, ' ', string)
        string = re.sub(r4, ' ', string)
        string = string.strip()
        string = self.remove_stopword_zh(string)
        return string

    def clean_str_BERT(self,string):
        r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符
        r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
        r3 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
        r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
        string = string.split('http')[0]
        cleanr = re.compile('<.*?>')
        string = re.sub(cleanr, ' ', string)
        string = re.sub(r4, ' ', string)
        return string

    def remove_stopword(self, string):
        if self.stop_words is None:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))

        if type(string) is str:
            string = string.split()

        new_string = list()
        for word in string:
            if word in self.stop_words:
                continue
            new_string.append(word)

        return " ".join(new_string)

    def remove_stopword_zh(self, string):
        stopwords = []
        with open('../data/weibo/stop_words.txt', 'r', encoding='utf-8')as f:
            txt = f.readlines()
        for line in txt:
            stopwords.append(line.strip('\n'))

        if type(string) is str:
            string = jieba.cut(string)

        new_string = list()
        for word in string:
            if word in stopwords:
                continue
            new_string.append(word)

        return " ".join(new_string)

    def replace_num(self, string):
        result = re.sub(self.num, '<num>', string)
        return result

    def replace_urls(self, string):
        result = re.sub(self.url, '<url>', string)
        result = ' '.join(re.split(' +|\n+', result)).strip()
        return result

class DataProcessor:
    """
           DataProcessor class to preprocess the data.

           Parameters:
               sen_len (int): Maximum length of sentences.
               pathset (object): Object containing paths to dataset files.
    """
    def __init__(self, sen_len, pathset):
        self.sen_len = sen_len
        self.pathset = pathset
        self.idx2node_dict = {}
        self.node2idx_dict = {}
        self.mid2text_dict = {}
        self.mid2bert_tokenizer = {}
        self.idx2user_dict={}
        self.train_dict={}
        self.val_dict={}
        self.test_dict={}
        self.root_id={}
        UNCASED = self.pathset.path_bert
        VOCAB = self.pathset.VOCAB
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))

    def get_node2id(self):
        with open(self.pathset.path_node2idx_mid, 'r', encoding='utf-8') as f:
            node2idx_mid = f.readlines()
        for line in node2idx_mid:
            node_idx = line.strip('\n').split('\t')
            node = node_idx[0]
            idx = int(node_idx[1])
            self.node2idx_dict[node] = idx
            self.idx2node_dict[idx] = node

    def get_mid2text(self):
        with open(self.pathset.path_mid2text, 'r', encoding='utf-8') as f:
            mid2text = f.readlines()
        for line in mid2text:
            mid_text = line.strip('\n').split('\t')
            mid = mid_text[0]
            text = mid_text[1]
            self.mid2text_dict[mid] = text

    def get_root_token(self, root_index):
        string_process = StringProcess()
        for idx in root_index:
            idx = int(idx)
            mid = self.idx2node_dict[idx]
            text = self.mid2text_dict[mid]
            text = string_process.clean_str_BERT(text)
            tokenizer_encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', \
                                                truncation=True, max_length=self.sen_len)
            self.mid2bert_tokenizer[idx] = tokenizer_encoding

    def get_idx2user(self):
        with open(self.pathset.path_mid2user, 'r', encoding='utf-8') as f:
            mid2user = f.readlines()
        for line in mid2user:
            mid_text = line.strip('\n').split('\t')
            mid = mid_text[0]
            user =[int(x) for x in mid_text[1:]]
            idx=self.node2idx_dict[mid]
            self.idx2user_dict[idx]=user

    def load_sparse_temporal_data(self, train, val, test):
        names = ['propagation_node_idx.npy', 'propagation_node.npy', 'label.npy', 'propagation_root_index.npy']
        objects = []
        for i in range(len(names)):
            with open(self.pathset.path_temporal + "/{}".format(names[i]), 'rb') as f:
                objects.append(np.load(f, encoding='latin1', allow_pickle=True))
        p_idx, p_node, y, p_root_idx = tuple(objects)
        # 去除只有一个节点的post（没有回复的）
        p_idx, p_node, y, p_root_idx = remove_single_nodepost(p_idx, p_node, y, p_root_idx)
        print('data process p_idx:', len(p_idx))
        print('data process p_root_idx:', len(p_root_idx))
        self.root_id=p_root_idx
        p_node_indices = []  #
        p_node_values = []
        # print("p_node.shape:", p_node.shape)  [事件数,时间分片数,[节点个数,节点个数]]
        for xx in p_node:
            xx_indices, xx_values = [], []
            for i in range(len(xx)):  # len(xx)==3  xx[i]为第i个事件段内的图结构(二维邻接矩阵)
                indices, values, shape = sparse_mx_to_torch(normalize_adj(xx[i]))  # indices 有边的(行,列)
                xx_indices.append(indices)
                xx_values.append(values)
            p_node_indices.append(xx_indices)  #
            p_node_values.append(xx_values)
        # -------------------------------------------------------------------------------------
        y = torch.from_numpy(y).long()
        y = torch.unsqueeze(y, 1)
        p_root_idx = [_idx.astype(int) for _idx in p_root_idx[:]]
        p_root_idx = np.array(p_root_idx)
        p_root_idx = torch.from_numpy(p_root_idx).long()
        p_root_idx = torch.unsqueeze(p_root_idx, 1)

        p_idx = [_idx.astype(int) for _idx in p_idx[:]]
        p_idx = [torch.from_numpy(_idx).long() for _idx in p_idx[:]]

        train_idx, val_idx, test_idx = split_data(len(p_idx), y, train, val, test, shuffle=True)

        names_dict = {'x_p_indices': p_node_indices, 'x_p_values': p_node_values, 'y': y, 'idx_p': p_idx,
                      'root_idx_p': p_root_idx}
        for name in names_dict:
            self.train_dict[name] = [names_dict[name][i] for i in train_idx]
            self.val_dict[name] = [names_dict[name][i] for i in val_idx]
            self.test_dict[name] = [names_dict[name][i] for i in test_idx]

    def getData(self,model_mode,train, val, test):
        self.get_node2id()
        self.get_mid2text()
        self.load_sparse_temporal_data(train, val, test)
        self.get_root_token(self.root_id)
        if model_mode == 'propagation':
            return self.train_dict, self.val_dict, self.test_dict, len(self.idx2node_dict), self.mid2bert_tokenizer
        elif model_mode == 'semantic':
            text_embedding = torch.tensor(np.load(self.pathset.text_embeddings_path))
            return self.train_dict, self.val_dict, self.test_dict,self.mid2bert_tokenizer,text_embedding
        elif model_mode =='user':
            self.get_idx2user()
            return self.train_dict, self.val_dict, self.test_dict, self.idx2user_dict, self.mid2bert_tokenizer
        elif model_mode in ['dual','wotf','static']:
            self.get_idx2user()
            text_embedding = torch.tensor(np.load(self.pathset.text_embeddings_path))
            return self.train_dict, self.val_dict, self.test_dict,self.idx2user_dict, self.mid2bert_tokenizer,text_embedding

#  Need to adjust the code according to the number of divisions N
class GraphDataset(Dataset):
    def __init__(self, x_propagation_idx,x_propagation_node_indices,x_propagation_node_values,target, root_idx):
        self.x_propagation_idx = x_propagation_idx
        self.x_propagation_node_indices = x_propagation_node_indices
        self.x_propagation_node_values = x_propagation_node_values
        self.target = target
        self.root_idx = root_idx
    def __len__(self):
        return len(self.target)
    def __getitem__(self, index):
        #==========================propogation graph==============================
        x_propagation_node_num = len(self.x_propagation_idx[index])
        x_propagation_node_idx = self.x_propagation_idx[index]
        x_propagation_node_idx = x_propagation_node_idx.unsqueeze(1)
        #--------------progation_graph_0----------------
        x_propagation_node_indices_0 = self.x_propagation_node_indices[index][0]
        x_p_edge_num_0 = self.x_propagation_node_indices[index][0].size(1)
        x_propagation_node_values_0 = self.x_propagation_node_values[index][0]
        x_propagation_node_values_0 = torch.tensor(x_propagation_node_values_0, dtype=torch.float)

        #-------------propagation_graph_1---------------
        x_propagation_node_indices_1 = self.x_propagation_node_indices[index][1]
        x_p_edge_num_1 = self.x_propagation_node_indices[index][1].size(1)
        x_propagation_node_values_1 = self.x_propagation_node_values[index][1]
        #-------------propagtion_graph_2----------------
        x_propagation_node_indices_2 = self.x_propagation_node_indices[index][2]
        x_p_edge_num_2 = self.x_propagation_node_indices[index][2].size(1)
        x_propagation_node_values_2 = self.x_propagation_node_values[index][2]
        #--------------------target--------------------
        if self.target[index].item() == 0:
            y = torch.tensor([1,0]).unsqueeze(0)
        elif self.target[index].item() == 1:
            y = torch.tensor([0,1]).unsqueeze(0)
        return Data(
                    x = torch.tensor(x_propagation_node_idx,dtype=torch.long),\
                    edge_index = torch.LongTensor(x_propagation_node_indices_0), \
                    edge_values=torch.tensor(x_propagation_node_values_0, dtype=torch.float), \
                    edge_num=torch.tensor([x_p_edge_num_0], dtype=torch.long), \

                    x_propagation_node_indices_1_edge_index= torch.LongTensor(x_propagation_node_indices_1), \
                    x_propagation_node_values_1=torch.tensor(x_propagation_node_values_1, dtype=torch.float), \
                    x_propagation_edge_num_1=torch.tensor([x_p_edge_num_1], dtype=torch.long), \

                    x_propagation_node_indices_2_edge_index = torch.LongTensor(x_propagation_node_indices_2), \
                    x_propagation_node_values_2 = torch.tensor(x_propagation_node_values_2,dtype=torch.float),\
                    x_propagation_edge_num_2 = torch.tensor([x_p_edge_num_2],dtype=torch.long), \

                    x_propagation_node_num = torch.tensor([x_propagation_node_num],dtype=torch.long),\

                    target = torch.tensor(y,dtype=torch.float),\
                    root_idx = torch.LongTensor(self.root_idx[index]))