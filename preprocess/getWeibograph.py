import os,re
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import sparse

# Paths to the data directories
weibo_clean_path = '../data/weibo/weibo_clean/'
weibo_temporal_path = '../data/weibo/weibo_temporal_data/'
max_node_num=200  # Set the maximum number of repost nodes

# Keys for different attributes
key = ['parent', 'text', 'bi_followers_count', 'friends_count', 'followers_count', 'statuses_count',
       'verified_type', 'favourites_count','reposts_count', 'comments_count', 'attitudes_count']

user_key=['bi_followers_count', 'friends_count', 'followers_count', 'statuses_count','verified',
          'verified_type', 'favourites_count','reg_len','gender','user_geo_enabled']
stat_key=['reposts_count', 'comments_count', 'attitudes_count']

# Function to clean the text data by removing certain characters
def clean_text(text):
    r1 = "\n"
    r2 = '\r'
    r3 = '\t'
    text = re.sub(r1,' ',text)
    text = re.sub(r2,' ',text)
    text = re.sub(r3,' ',text)
    return text
# Function to read data from files and process it into a dictionary
def read_data():
    tree_dic = {}
    pheme_files_entity = os.listdir(weibo_clean_path)
    for i in range(len(pheme_files_entity)):
        file = pheme_files_entity[i].split('.')[0]
        tree_dic[file] = {}
        file_df_entity = pd.read_csv(weibo_clean_path + file + '.csv')
        init_time =  file_df_entity[pd.isna(file_df_entity['parent'])]['t'].iloc[0]
        node_num=len(file_df_entity['mid'])
        num=max_node_num if max_node_num<node_num else node_num

        for j in range(num):
            mid = file_df_entity['mid'][j]
            if not mid in tree_dic[file]:
                tree_dic[file][mid] = {}
                for k in key:
                    tree_dic[file][mid][k]=file_df_entity[k][j]
                tree_dic[file][mid]['text'] = clean_text(str(tree_dic[file][mid]['text']))
                post_time=file_df_entity['t'][j]
                tree_dic[file][mid]['t']=post_time-init_time
                tree_dic[file][mid]['reg_len'] = ( post_time-file_df_entity['user_created_at'][j])//86400  #86400是一天的秒数
                verified= file_df_entity['verified'][j]
                gender=file_df_entity['gender'][j]
                user_geo_enabled = file_df_entity['user_geo_enabled'][j]
                tree_dic[file][mid]['gender'] = 1 if gender == 'f' else 0
                tree_dic[file][mid]['verified']= 1 if verified else 0
                tree_dic[file][mid]['user_geo_enabled']=1 if user_geo_enabled else 0
    return tree_dic

#Function to divide time into equal segments: N = 3
def time_equal_segment(sub_tree_dic):
    t_list = []
    for mid in sub_tree_dic:
        t = sub_tree_dic[mid]['t']
        t_list.append(t)
    max_t = max(t_list)
    sliding_T = int(max_t/3)
    T_num = 3
    return sliding_T, T_num

# Function to map nodes to indices
def node2index():
    tree_dic = read_data()
    files_name = [file.split('.')[0] for file in os.listdir(weibo_clean_path)]
    node_lst = []
    for file in files_name:
        for mid in tree_dic[file]:
            node_lst.append(str(mid))
    node_lst = list(set(node_lst))
    with open('../data/weibo/' + 'node2idx_mid.txt', 'w', encoding='utf-8', newline='')as f:
        for i, node in enumerate(node_lst):
            string = node + '\t' + str(i) + '\n'
            f.writelines(string)  #mid2text.txt contains some duplicate mids
    with open('../data/weibo/'+'mid2text.txt','w',encoding='utf-8',newline='')as f:
        for file in files_name:
            for mid in tree_dic[file]:
                string = str(mid) +'\t' + str(tree_dic[file][mid]['text']) + '\n'
                f.writelines(string)
    with open('../data/weibo/' + 'mid2user.txt', 'w', encoding='utf-8', newline='') as f:
        for file in files_name:
            for mid in tree_dic[file]:
                string = str(mid)
                for k in user_key:
                    string+='\t'+str(tree_dic[file][mid][k])
                string+='\n'
                f.writelines(string)
    with open('../data/weibo/' + 'mid2stat.txt', 'w', encoding='utf-8', newline='') as f:
        for file in files_name:
            for mid in tree_dic[file]:
                string = str(mid)
                for k in stat_key:
                    string+='\t'+str(tree_dic[file][mid][k])
                string+='\n'
                f.writelines(string)

# Function to load the node-to-index mapping from file
def load_node2index():
    with open('../data/weibo/' + 'node2idx_mid.txt', 'r',encoding='utf-8')as f:
        node2idx = f.readlines()
    node2idx_dict = {}
    for line in node2idx:
        node = line.strip('\n').split('\t')[0]
        idx = line.strip('\n').split('\t')[1]
        node2idx_dict[node] = idx
    return node2idx_dict

# Function to process the node-to-index mapping for a subtree
def process_node2index(node2idx, sub_tree_dic):
    text_node_lst = []
    for mid in sub_tree_dic:
        mid_new = str(mid)
        text_node_lst.append(node2idx[mid_new])
    return text_node_lst

# Function to load the mid-to-label mapping from file
def load_mid2label():
    label_path = '../data/weibo/weibo_id_label.txt'
    mid_label = list(open(label_path, "r",encoding='utf-8'))
    mid2label_dict = {}
    for i, m_l in enumerate(mid_label):
        l = m_l.strip('\n').split('\t')
        mid = l[0]
        label =int(l[1])
        mid2label_dict[mid] = label
    return mid2label_dict

# Function to build the temporal propagation graph. Need to adjust the code according to the number of divisions N
def build_temporal_propagation_graph(text_node_idx,node2idx,sub_tree_dic):
    length = len(text_node_idx)
    sliding_T, T_num = time_equal_segment(sub_tree_dic)
    temporal_matrix = np.zeros((T_num,length,length),dtype=np.float)
    for mid in sub_tree_dic:
        if sub_tree_dic[mid]['t'] < sliding_T:
            idx = node2idx[str(mid)]
            new_idx = text_node_idx.index(idx)
            if str(sub_tree_dic[mid]['parent']) in node2idx:
                parent_idx = node2idx[str(sub_tree_dic[mid]['parent'])]
                new_parent_idx = text_node_idx.index(parent_idx)
                assert new_idx != new_parent_idx
                temporal_matrix[0][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[0][new_parent_idx][new_idx] = float(1.0)
                temporal_matrix[1][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[1][new_parent_idx][new_idx] = float(1.0)
                temporal_matrix[2][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[2][new_parent_idx][new_idx] = float(1.0)

        elif sub_tree_dic[mid]['t'] < 2*sliding_T and sub_tree_dic[mid]['t'] >= sliding_T:
            idx = node2idx[str(mid)]
            new_idx = text_node_idx.index(idx)
            if str(sub_tree_dic[mid]['parent']) in node2idx:
                parent_idx = node2idx[str(sub_tree_dic[mid]['parent'])]
                new_parent_idx = text_node_idx.index(parent_idx)
                assert new_idx != new_parent_idx
                temporal_matrix[1][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[1][new_parent_idx][new_idx] = float(1.0)
                temporal_matrix[2][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[2][new_parent_idx][new_idx] = float(1.0)

        else:
            idx = node2idx[str(mid)]
            new_idx = text_node_idx.index(idx)
            if str(sub_tree_dic[mid]['parent']) in node2idx:
                parent_idx = node2idx[str(sub_tree_dic[mid]['parent'])]
                new_parent_idx = text_node_idx.index(parent_idx)
                assert new_idx != new_parent_idx
                temporal_matrix[2][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[2][new_parent_idx][new_idx] = float(1.0)

    return temporal_matrix

def main():
    """
      Main function to load mappings, read data, process nodes, and build temporal propagation graphs.
      Need to adjust the code according to the number of divisions N
    """
    node2idx_dict = load_node2index()
    mid2label_dict = load_mid2label()
    tree_dic = read_data()
    files_name = [file.split('.')[0] for file in os.listdir(weibo_clean_path)]
    text_node_idx_final = []
    temp_propagation_graph_final = []
    label_final = []
    root_index_final = []
    for file in tqdm(files_name):
        if file == '3495745049431351': #该文件有问题，父节点id不存在该列表中
            continue
        else:
            for mid in tree_dic[file]:
                if pd.isna(tree_dic[file][mid]['parent']):
                    root_idx = node2idx_dict[str(mid)]
                    root_index_final.append(root_idx)
                    break
            label_final.append(mid2label_dict[file])
            text_node_idx = process_node2index(node2idx_dict,tree_dic[file])
            try:
                assert root_idx in text_node_idx
            except:
                with open('error_file_rootindex.txt','w',encoding='utf-8',newline='')as f:
                    string = file+'\t'+root_idx+'\n'
                    f.writelines(string)

            try:
                temp_propagation_graph = build_temporal_propagation_graph(text_node_idx,node2idx_dict,tree_dic[file])
            except ValueError:
                with open('error_file_propagation.txt','w',encoding='utf-8',newline='')as f:
                    string = file+'\t'+text_node_idx+'\n'
                    f.writelines(string)

            #print('propagation graph',temp_propagation_graph.shape)

            text_node_idx = np.array(text_node_idx)
            text_node_idx_final.append(text_node_idx)
            propagation_s0 = sparse.csr_matrix(temp_propagation_graph[0])
            propagation_s1 = sparse.csr_matrix(temp_propagation_graph[1])
            propagation_s2 = sparse.csr_matrix(temp_propagation_graph[2])
            temp_propagation_graph_final.append([propagation_s0,propagation_s1,propagation_s2])

    with open(weibo_temporal_path + 'propagation_node_idx.npy', 'wb')as f: #[[idx,idx,...],[idx,idx,...]] n*节点个数
        text_node_idx_final = np.array(text_node_idx_final)
        np.save(f,text_node_idx_final)
    with open(weibo_temporal_path + 'propagation_node.npy', 'wb')as f:  #[[p_s0,p_s1,p_s2],...],邻接矩阵元素为idx在list中的索引 n*3*邻接矩阵
        temp_propagation_graph_final = np.array(temp_propagation_graph_final)
        np.save(f, temp_propagation_graph_final)
    with open(weibo_temporal_path + 'label.npy', 'wb')as f:  #[0,1,1,0,.....] n*1
        label_final = np.array(label_final)
        np.save(f, label_final)
    with open(weibo_temporal_path + 'propagation_root_index.npy', 'wb')as f:  #[idx,idx,.....] node2idx_mid.txt mid的序列编号 n*1
        root_index_final = np.array(root_index_final)
        np.save(f, root_index_final)

if __name__ == '__main__':
    # Create node-to-index mappings and save to files
    node2index()
    # Run the main processing function
    main()

