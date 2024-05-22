import os
import pandas as pd
import datetime as dt
import re
import numpy as np
from tqdm import tqdm
from scipy import sparse
# Path to the cleaned CSV files containing initial data
pheme_clean_path = '../data/pheme/pheme_clean/'
# Path to save the processed numpy files
pheme_temporal_path = '../data/pheme/pheme_temporal_data/'

#Keys representing various features of the posts
key = ['parent', 'text', 'friends_count', 'followers_count', 'statuses_count', 'favourites_count',
       'listed_count', 'reposts_count', 'attitudes_count']
# Keys representing user-specific features
user_key=['friends_count', 'followers_count', 'statuses_count','verified', 'favourites_count',
          'listed_count', 'reg_len', 'following','user_geo_enabled', 'protected']
# Keys representing statistical features of the posts
stat_key=['reposts_count', 'attitudes_count']


def month_trans(mon):
    """
        Converts month abbreviation to month number.
    """
    mon_dic = {}
    mon_dic['Jan'] = 1
    mon_dic['Feb'] = 2
    mon_dic['Mar'] = 3
    mon_dic['Apr'] = 4
    mon_dic['May'] = 5
    mon_dic['Jun'] = 6
    mon_dic['Jul'] = 7
    mon_dic['Aug'] = 8
    mon_dic['Sep'] = 9
    mon_dic['Oct'] = 10
    mon_dic['Nov'] = 11
    mon_dic['Dec'] = 12
    return mon_dic[mon]

def trans_time(t, t_init):
    """
        Converts time strings into seconds since a reference time (t_init).
    """
    t = t.split(' ')
    t_exct = t[3].split(':')
    t_init = t_init.split(' ')
    t_init_exct = t_init[3].split(':')
    date_1 = dt.datetime(int(t[5]),month_trans(t[1]),int(t[2]),int(t_exct[0]),int(t_exct[1]),int(t_exct[2]))
    date_0 = dt.datetime(int(t_init[5]), month_trans(t_init[1]), int(t_init[2]), int(t_init_exct[0]), int(t_init_exct[1]), int(t_init_exct[2]))
    interval = (date_1 - date_0).seconds
    return interval

def clean_text(text):
    """
        Cleans the text by removing newlines, carriage returns, and tabs.
    """
    r1 = "\n"
    r2 = '\r'
    r3 = '\t'
    text = re.sub(r1,' ',text)
    text = re.sub(r2,' ',text)
    text = re.sub(r3,' ',text)
    return text


def read_data():
    """
    Reads the CSV files and constructs a dictionary representing the tree structure of each post.
    """
    tree_dic = {}
    pheme_files_entity = os.listdir(pheme_clean_path)
    for i in range(len(pheme_files_entity)):
        file = pheme_files_entity[i].split('.')[0]
        tree_dic[file] = {}
        file_df_entity = pd.read_csv(pheme_clean_path + file + '.csv')

        t_init = file_df_entity[pd.isna(file_df_entity['parent'])]['t'].iloc[0]

        for j in range(len(file_df_entity['mid'])):
            mid = file_df_entity['mid'][j]
            if not mid in tree_dic[file]:
                tree_dic[file][mid] = {}
                for k in key:
                    tree_dic[file][mid][k] = file_df_entity[k][j]

                tree_dic[file][mid]['text'] = clean_text(str(tree_dic[file][mid]['text']))
                post_time = file_df_entity['t'][j]
                t_trans = trans_time(post_time, t_init)
                tree_dic[file][mid]['t'] = t_trans
                tree_dic[file][mid]['reg_len'] = trans_time(post_time,file_df_entity['user_created_at'][j]) // 86400  # 86400是一天的秒数
                verified = file_df_entity['verified'][j]
                following=file_df_entity['following'][j]
                user_geo_enabled = file_df_entity['user_geo_enabled'][j]
                protected=file_df_entity['protected'][j]

                tree_dic[file][mid]['verified'] = 1 if verified else 0
                tree_dic[file][mid]['following'] = 1 if following else 0
                tree_dic[file][mid]['user_geo_enabled'] = 1 if user_geo_enabled else 0
                tree_dic[file][mid]['protected'] = 1 if protected else 0

    return tree_dic

#--------------comment segment---------------
def time_equal_segment(sub_tree_dic):
    """
        Segments the comments by time into three equal parts， N=1,2,3,4,5,6
    """
    t_list = []
    for mid in sub_tree_dic:
        t = sub_tree_dic[mid]['t']
        t_list.append(t)
    max_t = max(t_list)
    sliding_T = max_t/3
    T_num = 3

    return sliding_T, T_num


def node2index():
    """
      Assigns a unique index to each node and saves the mappings to text files.
    """
    tree_dic = read_data()
    files_name = [file.split('.')[0] for file in os.listdir(pheme_clean_path)]
    node_lst = []

    for file in files_name:
        for mid in tree_dic[file]:
            node_lst.append(str(mid))
    node_lst = list(set(node_lst))
    with open('../data/pheme/' + 'node2idx_mid.txt', 'w', encoding='utf-8', newline='')as f:
        for i, node in enumerate(node_lst):
            string = node + '\t' + str(i) + '\n'
            f.writelines(string)
    with open('../data/pheme/'+'mid2text.txt','w',encoding='utf-8',newline='')as f:
        for file in files_name:
            for mid in tree_dic[file]:
                string = str(mid) +'\t' + tree_dic[file][mid]['text'] + '\n'
                f.writelines(string)
    with open('../data/pheme/' + 'mid2user.txt', 'w', encoding='utf-8', newline='') as f:
        for file in files_name:
            for mid in tree_dic[file]:
                string = str(mid)
                for k in user_key:
                    string+='\t'+str(tree_dic[file][mid][k])
                string+='\n'
                f.writelines(string)
    with open('../data/pheme/' + 'mid2stat.txt', 'w', encoding='utf-8', newline='') as f:
        for file in files_name:
            for mid in tree_dic[file]:
                string = str(mid)
                for k in stat_key:
                    string+='\t'+str(tree_dic[file][mid][k])
                string+='\n'
                f.writelines(string)

def load_node2index():
    """
        Loads the node-to-index mapping from a text file into a dictionary.
    """
    with open('../data/pheme/' + 'node2idx_mid.txt', 'r',encoding='utf-8')as f:
        node2idx = f.readlines()
    node2idx_dict = {}
    for line in node2idx:
        node = line.strip('\n').split('\t')[0]
        idx = line.strip('\n').split('\t')[1]
        node2idx_dict[node] = idx
    return node2idx_dict

def process_node2index(node2idx, sub_tree_dic):
    """
     Converts node identifiers in the subtree dictionary to their corresponding indices.
    """
    text_node_lst = []
    for mid in sub_tree_dic:
        mid_new = str(mid)
        text_node_lst.append(node2idx[mid_new])

    return text_node_lst

def load_mid2label():
    """
    Loads the post-to-label mapping from a text file into a dictionary.
    """
    label_path = '../data/pheme/pheme_id_label.txt'
    mid_label = list(open(label_path, "r",encoding='utf-8'))
    mid2label_dict = {}
    for m_l in mid_label:
        l = m_l.strip('\n').split('\t')
        mid = l[0]
        label = int(l[1])
        mid2label_dict[mid] = int(label)
    return mid2label_dict


def build_temporal_propagation_graph(text_node_idx,node2idx,sub_tree_dic):
    """
    Constructs a temporal propagation graph based on the segmented time intervals.
    Need to adjust the code according to the number of divisions N
    """
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
    files_name = [file.split('.')[0] for file in os.listdir(pheme_clean_path)]
    text_node_idx_final = []
    temp_propagation_graph_final = []
    label_final = []
    root_index_final = []
    for file in tqdm(files_name):
        root_index_final.append(node2idx_dict[file])
        label_final.append(mid2label_dict[file])
        text_node_idx = process_node2index(node2idx_dict,tree_dic[file])
        # -------------------传播图------------------------
        temp_propagation_graph = build_temporal_propagation_graph(text_node_idx,node2idx_dict,tree_dic[file])

        text_node_idx = np.array(text_node_idx)
        text_node_idx_final.append(text_node_idx)
        propagation_s0 = sparse.csr_matrix(temp_propagation_graph[0])
        propagation_s1 = sparse.csr_matrix(temp_propagation_graph[1])
        propagation_s2 = sparse.csr_matrix(temp_propagation_graph[2])
        temp_propagation_graph_final.append([propagation_s0,propagation_s1,propagation_s2])


    with open(pheme_temporal_path+'propagation_node_idx.npy','wb')as f:
        text_node_idx_final = np.array(text_node_idx_final)
        np.save(f,text_node_idx_final)
    with open(pheme_temporal_path+'propagation_node.npy','wb')as f:
        temp_propagation_graph_final = np.array(temp_propagation_graph_final)
        np.save(f, temp_propagation_graph_final)
    with open(pheme_temporal_path+'label.npy','wb')as f:
        label_final = np.array(label_final)
        np.save(f, label_final)
    with open(pheme_temporal_path+'propagation_root_index.npy','wb')as f:
        root_index_final = np.array(root_index_final)
        np.save(f, root_index_final)


if __name__ == '__main__':
    # Create node-to-index mappings and save to files
    node2index()
    # Run the main processing function
    main()



