import os

class path_Set():
    def __init__(self,dataset):
        self.path_data_dir = os.path.join("../", "data/{}".format(dataset))

        self.path_node2idx_mid = os.path.join(self.path_data_dir, "node2idx_mid.txt")
        self.path_mid2text = os.path.join(self.path_data_dir, 'mid2text.txt')
        self.path_mid2user=os.path.join(self.path_data_dir,'mid2user.txt')
        self.path_mid2stat=os.path.join(self.path_data_dir,'mid2stat.txt')

        # data
        self.path_temporal = os.path.join(self.path_data_dir, "{}_temporal_data".format(dataset))
        self.path_temporal_propagation_idx = os.path.join(self.path_data_dir, "{}_temporal_data/propagation_node_idx.npy".format(dataset))
        self.path_temporal_propagation_graph = os.path.join(self.path_data_dir, "{}_temporal_data/propagation_node.npy".format(dataset))
        self.path_label = os.path.join(self.path_data_dir, "{}_temporal_data/label.npy".format(dataset))
        self.path_root_idx = os.path.join(self.path_data_dir, "{}_temporal_data/propagation_root_index.npy".format(dataset))
        self.text_embeddings_path=os.path.join(self.path_data_dir,'{}_temporal_data/text_embeddings.npy'.format(dataset))

        #trained_model
        self.path_saved_model = "model_saved/"
        #BERT_PATH
        if dataset == 'weibo':
            self.path_bert = './bert-base-chinese/'
            self.VOCAB = 'vocab.txt'
        elif dataset == 'pheme':
            self.path_bert = './bert-base-uncased/'
            self.VOCAB = 'vocab.txt'