import json
import pickle
import scipy.sparse as sp
import numpy as np
import torch
from numpy.linalg import norm
from tqdm import tqdm
from collections import defaultdict


def feature_process(feature_a):
    feature_a = feature_a.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((feature_a.row, feature_a.col)).astype(np.int64))  # 行列索引合并
    values = torch.from_numpy(feature_a.data)  # value值
    shape = torch.Size(feature_a.shape)  # 特征矩阵的大小
    return torch.sparse.FloatTensor(indices, values, shape)


def KNN(top_k, adj):
    indices = np.argpartition(-adj, top_k, axis=1)[:, :top_k].tolist()
    row = []
    col = []
    value = []
    empty = {}
    num = 0
    for i, j in enumerate(indices):
        for k in j:
            if adj[i][k] > 0:
                row.append(i)
                col.append(k)
                value.append(1)
            else:
                num += 1
    new_adj = sp.coo_matrix((value, (row, col)), shape=adj.shape)
    return new_adj


def KNN(top_k, adj):
    indices = np.argpartition(-adj, top_k, axis=1)[:, :top_k].tolist()
    row = []
    col = []
    value = []
    empty = {}
    num = 0
    for i, j in enumerate(indices):
        for k in j:
            if adj[i][k] > 0:
                row.append(i)
                col.append(k)
                value.append(1)
            else:
                num += 1
    new_adj = sp.coo_matrix((value, (row, col)), shape=adj.shape)
    return new_adj


def create_meta_graph(meta, relation_list, s=1):
    meta_adj = relation_list[meta[0]].to_dense().numpy()
    name = meta[0]
    for i in meta[1:]:
        present_adj = relation_list[i].to_dense().numpy()
        meta_adj = np.matmul(meta_adj, present_adj)
        name = name[:-1] + i
    indices = np.nonzero(meta_adj)
    row = []
    col = []
    value = []
    for i, j in zip(indices[0].tolist(), indices[1].tolist()):
        if meta_adj[i][j] >= s:
            row.append(i)
            col.append(j)
            value.append(1)
    new_adj = sp.coo_matrix((value, (row, col)), shape=meta_adj.shape)
    # with open('../data/2024_2_08/dataset/relation_edge/meta_{}.plk'.format(name), 'wb') as gf:
    #     pickle.dump(new_adj, gf)
    return new_adj, name, sp.coo_matrix(meta_adj)


# 超过两条的
def create_meta_more_graph(meta):
    meta_adj = meta.to_dense().numpy()
    new_meta = np.matmul(meta_adj, meta_adj.T)
    indices = np.nonzero(new_meta)
    row = []
    col = []
    value = []
    for i, j in zip(indices[0].tolist(), indices[1].tolist()):
        row.append(i)
        col.append(j)
        value.append(1)
    new_adj = sp.coo_matrix((value, (row, col)), shape=meta_adj.shape)
    # with open('../data/2024_2_08/dataset/relation_edge/meta_{}.plk'.format(name), 'wb') as gf:
    #     pickle.dump(new_adj, gf)
    return new_adj, sp.coo_matrix(new_meta)


# 元图处理
def create_metagraph_graph(meta1, meta2):
    meta1_adj = meta1.to_dense().numpy()
    meta2_adj = meta2.to_dense().numpy()
    mete_graph_adj = meta1_adj * meta2_adj
    indices = np.nonzero(mete_graph_adj)
    row = []
    col = []
    value = []
    for i, j in zip(indices[0].tolist(), indices[1].tolist()):
        row.append(i)
        col.append(j)
        value.append(1)
    new_adj = sp.coo_matrix((value, (row, col)), shape=mete_graph_adj.shape)
    return new_adj


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj = sp.coo_matrix(adj)  # 4057, 4057
    rowsum = np.array(adj.sum(1))  # 4057， 统计每个源路径数量 1
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # 4057， 1
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# def heter_normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     # adj = sp.coo_matrix(adj)  # 4057, 4057
#     rowsum = np.array(adj.sum(1))   # 4057， 统计每个源路径数量 1\
#     a = np.repeat(rowsum, 3567, 1)
#     d_mat_inv_sqrt = sp.coo_matrix(a)
#     return d_mat_inv_sqrt * adj     # .tocoo()
# 基因-基因 4004464

def select_homo_neigh(adj):
    new_adj = np.where(adj > 0.5, 1, 0)
    num = new_adj.sum()
    print('边数: {}'.format(num))
    return sp.coo_matrix(new_adj)


def select_homo_neigh_c(adj, n):
    new_adj = np.where(adj > n, 1, 0)
    num = new_adj.sum()
    print('边数: {}'.format(num))
    return sp.coo_matrix(new_adj)


def select_sem_neigh(adj, adj2):
    adj = feature_process(adj)
    # adj2 = feature_process(adj2)
    adj = adj.to_dense().numpy()
    # adj2 = adj2.to_dense().numpy()
    new_adj = np.where(adj > 0, adj2, 0)
    # num = new_adj.sum()
    # print('边数: {}'.format(num))
    return sp.coo_matrix(new_adj)


def cos(A, B):
    cosine = np.dot(A, B) / (norm(A) * norm(B))
    # print("余弦相似度:", cosine)
    return cosine


# 基于结构的
# 30791  2 2591 3 4219 10 9789

def non_zero_mean(np_arr, axis=1):
    """ndarray按行/列求非零元素的均值。
    axis=0按列
    axis=1按行
    """
    exist = (np_arr != 0)
    num = np_arr.sum(axis=axis)
    den = exist.sum(axis=axis)
    return num / den


def sample_meta_path_method(adj):
    adj = adj.to_dense().numpy()
    b = adj.sum()
    a = non_zero_mean(adj)
    new_adj = np.zeros_like(adj)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] >= a[i]:
                new_adj[i][j] = 1
    c = new_adj.sum()
    print('边数: {}'.format(new_adj.sum()))
    return sp.coo_matrix(new_adj)


def sample_meta_path_method_c(adj, multiply):
    adj = adj.to_dense().numpy()
    multiply = multiply.to_dense().numpy()
    a = adj.sum()
    threshold = np.sum(adj, axis=1)
    multiply_c = np.sum(multiply, axis=1)
    new_adj = np.zeros_like(adj)
    for n, i in enumerate(threshold):
        if multiply_c[n] > 0:
            t = int(i / multiply_c[n])
        else:
            t = int(i)
        for nn, k in enumerate(adj[n]):
            if k > t and k > 0:
                new_adj[n][nn] = float(1)
    print('边数: {}'.format(new_adj.sum()))
    return sp.coo_matrix(new_adj)


# 相似度采样方法（语义）
def sample_meta_sem_method(f1, adj):
    adj = adj.to_dense().numpy()
    new_adj = np.zeros_like(adj)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            new_adj[i][j] = float(cos(f1[i], f1[j]))
    # new_adj = new_adj * adj
    with open('../../data/2024_2_08/dataset/relation_edge/adj_gg_sim.plk', 'wb') as ggs:
        pickle.dump(new_adj, ggs)
    print(1)
    return sp.coo_matrix(new_adj)


def load_data(r, args):
    # 加载特征
    with open('../data/dataset1/node_features/gene_pac_feature_new_512.npy', 'rb') as gf:
        gene_features = np.load(gf)
    with open('../data/dataset1/node_features/hpo_pac_feature_512.npy', 'rb') as hf:
        hpo_features = np.load(hf)
    with open('../data/dataset1/node_features/disease_pac_feature_new_512.npy', 'rb') as df:
        disease_features = np.load(df)
    # 基因关系矩阵加载
    with open('../data/dataset1/relation_edge/adj_gg.plk', 'rb') as gg:
        adj_gg = pickle.load(gg)
    with open('../data/dataset1/relation_edge/adj_gh.plk', 'rb') as gh:
        adj_gh = pickle.load(gh)
    with open('../data/dataset1/relation_edge/adj_gd_{}.plk'.format(r), 'rb') as gd:
        adj_gd = pickle.load(gd)
        # 表型关系矩阵加载
    with open('../../data/2024_2_08/dataset/relation_edge/adj_hg.plk', 'rb') as hg:
        adj_hg = pickle.load(hg)
    with open('../../data/2024_2_08/dataset/relation_edge/adj_hh.plk', 'rb') as hh:
        adj_hh = pickle.load(hh)
    with open('../../data/2024_2_08/dataset/relation_edge/adj_hd.plk', 'rb') as hd:
        adj_hd = pickle.load(hd)
        # 表型关系矩阵加载
    with open('../data/2024_2_08/dataset/relation_edge/adj_dg_{}.plk'.format(r), 'rb') as dg:
        adj_dg = pickle.load(dg)
    with open('../../data/2024_2_08/dataset/relation_edge/adj_dh.plk', 'rb') as dh:
        adj_dh = pickle.load(dh)
    with open('../../data/2024_2_08/dataset/relation_edge/adj_dd.plk', 'rb') as dd:
        adj_dd = pickle.load(dd)
    # k近邻选取邻居
    new_adj_gg = select_homo_neigh(feature_process(adj_gg).to_dense().numpy())
    new_adj_hh = select_homo_neigh(feature_process(adj_hh).to_dense().numpy())
    new_adj_dd = select_homo_neigh(feature_process(adj_dd).to_dense().numpy())


    features_set = [gene_features, hpo_features, disease_features]

    with open('../data/2024_2_08/dataset/fold{}/adj_meta_ghg.plk'.format(r), 'rb') as mt:
        adj_meta_ghg = pickle.load(mt)
    with open('../data/2024_2_08/dataset/fold{}/adj_meta_gdg.plk'.format(r), 'rb') as mtd:
        adj_meta_gdg = pickle.load(mtd)
    with open('../data/2024_2_08/dataset/fold{}/adj_meta_dhd.plk'.format(r), 'rb') as dhd:
        adj_meta_dhd = pickle.load(dhd)
    with open('../data/2024_2_08/dataset/fold{}/adj_meta_dgd.plk'.format(r), 'rb') as dgd:
        adj_meta_dgd = pickle.load(dgd)
    print(1)

    meta_path_list = {
        'g': {
            'gg': [feature_process(new_adj_gg)],
            'gog': [feature_process(adj_meta_ghg)],  # feature_process(normalize_adj(ghg)),
            'gdg': [feature_process(adj_meta_gdg)]},
        'd': {
            'dd': [feature_process(new_adj_dd)],
            'dhd': [feature_process(adj_meta_dhd)],
            'dgd': [feature_process(adj_meta_dgd)],
        }
    }

    with open('../data/2024_2_08/dataset/sample/fold_{}.json'.format(r), 'r') as s:
        sample = json.load(s)

    train_sample = np.array(sample[0])
    test_sample = np.array(sample[1])
    train_target_gene_index = train_sample[:, 0]
    train_target_hpo_index = train_sample[:, 1]

    train_label = train_sample[:, 2]
    train_sample_list = [train_target_gene_index, train_target_hpo_index, train_label]
    test_target_gene_index = test_sample[:, 0]
    test_target_hpo_index = test_sample[:, 1]
    test_label = test_sample[:, 2]
    test_sample_list = [test_target_gene_index, test_target_hpo_index, test_label]
    print(len(train_target_gene_index))
    print(len(test_target_gene_index))
    pos_inter = [feature_process(adj_gg), feature_process(adj_gg)]
    pos_outer = [feature_process(sp.coo_matrix(np.diag(np.ones(4454)))),
                 feature_process(sp.coo_matrix(np.diag(np.ones(3567))))]
    print('g_num:{}'.format(len(meta_path_list['g'])))
    print('d_num:{}'.format(len(meta_path_list['d'])))
    return features_set, meta_path_list, train_sample_list, test_sample_list, pos_inter, pos_outer


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience  # 30
        self.verbose = verbose  # Ture
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss
