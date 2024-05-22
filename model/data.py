import numpy as np
import scipy.sparse as sp
import torch
import random

def split_data(size,y, train, val, test, shuffle=True):
    # Split the data into train, validation, and test sets
    idx = list(range(size))
    idx_pos = []
    idx_neg = []
    for idx_temp in idx:
        # label_dict[idx_temp] = y[idx_temp].item()
        if y[idx_temp].item() == 1:
            idx_neg.append(idx_temp)
        elif y[idx_temp].item() == 0:
            idx_pos.append(idx_temp)
    print('Data distribution：','pos:',len(idx_pos),'neg',len(idx_neg))
    if shuffle:
        np.random.shuffle(idx_neg)
        np.random.shuffle(idx_pos)
    split_idx_pos = np.split(idx_pos, [int(train * len(idx_pos)), int((train + val) * len(idx_pos))])
    train_idx_pos, val_idx_pos, test_idx_pos = split_idx_pos[0], split_idx_pos[1], split_idx_pos[2]

    split_idx_neg = np.split(idx_neg, [int(train * len(idx_neg)), int((train + val) * len(idx_neg))])
    train_idx_neg, val_idx_neg, test_idx_neg = split_idx_neg[0], split_idx_neg[1], split_idx_neg[2]

    train_idx = np.concatenate((train_idx_pos, train_idx_neg),axis=0)
    val_idx = np.concatenate((val_idx_pos,val_idx_neg),axis=0)
    test_idx = np.concatenate((test_idx_pos,test_idx_neg),axis=0)

    print('Train data distribution：', 'pos:', len(train_idx_pos), 'neg', len(train_idx_neg))
    print('Validation data distribution：', 'pos:', len(val_idx_pos), 'neg', len(val_idx_neg))
    print('Test data distribution：', 'pos:', len(test_idx_pos), 'neg', len(test_idx_neg))
    return train_idx, val_idx, test_idx

def split_data_5fold(size,y, train, val, test, shuffle=True):
    # Split the data into 5 folds for cross-validation
    idx = list(range(size))
    idx_pos = []
    idx_neg = []
    for idx_temp in idx:
        # label_dict[idx_temp] = y[idx_temp].item()
        if y[idx_temp].item() == 1:
            idx_neg.append(idx_temp)
        elif y[idx_temp].item() == 0:
            idx_pos.append(idx_temp)
    print('Data distribution：','pos:',len(idx_pos),'neg',len(idx_neg))
    if shuffle:
        random.shuffle(idx_neg)
        random.shuffle(idx_pos)
    fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
    fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
    fold0_x_val, fold1_x_val, fold2_x_val, fold3_x_val, fold4_x_val = [], [], [], [], []
    leng1 = int(len(idx_pos) * test)
    leng2 = int(len(idx_neg) * test)



    fold0_x_test.extend(idx_pos[0:leng1])
    fold0_x_test.extend(idx_neg[0:leng2])
    temp_pos = idx_pos-idx_pos[0:leng1]
    temp_neg = idx_neg-idx_neg[0:leng2]
    leng3 = int(len(temp_pos) * val)
    leng4 = int(len(temp_neg) * val)
    fold0_x_val.extend(temp_pos[0:leng3])
    fold0_x_val.extend(temp_neg[0:leng4])
    fold0_x_train.extend(temp_pos[leng3:])
    fold0_x_train.extend(temp_neg[leng4:])

    fold1_x_test.extend(idx_pos[leng1:leng1*2])
    fold1_x_test.extend(idx_neg[leng2:leng2*2])
    temp_pos = idx_pos - idx_pos[leng1:leng1*2]
    temp_neg = idx_neg - idx_neg[leng2:leng2*2]
    leng3 = int(len(temp_pos) * val)
    leng4 = int(len(temp_neg) * val)
    fold1_x_val.extend(temp_pos[0:leng3])
    fold1_x_val.extend(temp_neg[0:leng4])
    fold1_x_train.extend(temp_pos[leng3:])
    fold1_x_train.extend(temp_neg[leng4:])

    fold2_x_test.extend(idx_pos[leng1*2:leng1*3])
    fold2_x_test.extend(idx_neg[leng2*2:leng2*3])
    temp_pos = idx_pos - idx_pos[leng1*2:leng1*3]
    temp_neg = idx_neg - idx_neg[leng2*2:leng2*3]
    leng3 = int(len(temp_pos) * val)
    leng4 = int(len(temp_neg) * val)
    fold2_x_val.extend(temp_pos[0:leng3])
    fold2_x_val.extend(temp_neg[0:leng4])
    fold2_x_train.extend(temp_pos[leng3:])
    fold2_x_train.extend(temp_neg[leng4:])

    fold3_x_test.extend(idx_pos[leng1*3:leng1*4])
    fold3_x_test.extend(idx_neg[leng2*3:leng2*4])
    temp_pos = idx_pos - idx_pos[leng1*3:leng1*4]
    temp_neg = idx_neg - idx_neg[leng2*3:leng2*4]
    leng3 = int(len(temp_pos) * val)
    leng4 = int(len(temp_neg) * val)
    fold3_x_val.extend(temp_pos[0:leng3])
    fold3_x_val.extend(temp_neg[0:leng4])
    fold3_x_train.extend(temp_pos[leng3:])
    fold3_x_train.extend(temp_neg[leng4:])

    fold4_x_test.extend(idx_pos[leng1*4:])
    fold4_x_test.extend(idx_neg[leng2*4:])
    temp_pos = idx_pos - idx_pos[leng1*4:]
    temp_neg = idx_neg - idx_neg[leng2*4:]
    leng3 = int(len(temp_pos) * val)
    leng4 = int(len(temp_neg) * val)
    fold4_x_val.extend(temp_pos[0:leng3])
    fold4_x_val.extend(temp_neg[0:leng4])
    fold4_x_train.extend(temp_pos[leng3:])
    fold4_x_train.extend(temp_neg[leng4:])

    fold0_test = list(fold0_x_test)
    random.shuffle(fold0_test)
    fold0_val = list(fold0_x_val)
    random.shuffle(fold0_val)
    fold0_train = list(fold0_x_train)
    random.shuffle(fold0_train)

    fold1_test = list(fold1_x_test)
    random.shuffle(fold1_test)
    fold1_val = list(fold1_x_val)
    random.shuffle(fold1_val)
    fold1_train = list(fold1_x_train)
    random.shuffle(fold1_train)

    fold2_test = list(fold2_x_test)
    random.shuffle(fold2_test)
    fold2_val = list(fold2_x_val)
    random.shuffle(fold2_val)
    fold2_train = list(fold2_x_train)
    random.shuffle(fold2_train)

    fold3_test = list(fold3_x_test)
    random.shuffle(fold3_test)
    fold3_val = list(fold3_x_val)
    random.shuffle(fold3_val)
    fold3_train = list(fold3_x_train)
    random.shuffle(fold3_train)

    fold4_test = list(fold4_x_test)
    random.shuffle(fold4_test)
    fold4_val = list(fold4_x_val)
    random.shuffle(fold4_val)
    fold4_train = list(fold4_x_train)
    random.shuffle(fold4_train)

    return list(fold0_test),list(fold0_val), list(fold0_train), \
           list(fold1_test),list(fold1_val), list(fold1_train), \
           list(fold2_test),list(fold2_val), list(fold2_train), \
           list(fold3_test),list(fold3_val), list(fold3_train), \
           list(fold4_test),list(fold4_val), list(fold4_train)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj += sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 or len(sparse_mx.col) == 0:
        print(sparse_mx.row, sparse_mx.col)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_mx_to_torch(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 or len(sparse_mx.col) == 0:
        print(sparse_mx.row, sparse_mx.col)
        print('data bug')
        print('sparse_mx.data',sparse_mx.data)
        print('sparse_mx.shape',sparse_mx.shape)
    if np.NAN in sparse_mx.data:
        print('NaN')
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return indices,values,shape

