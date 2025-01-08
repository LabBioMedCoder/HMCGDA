import torch
import torch.nn as nn
import torch.nn.functional as F
# from meta_model import *
# from neigh_model import *
# from meta_model_3 import *


from meta_mode_feature_create_graph import *


# from meta5 import *


class overall_model(nn.Module):
    def __init__(self,
                 homo_meta_gene_num, homo_meta_dis_num, meta_layer_num,  # 元路径那块参数
                 neigh_layer_num, hidden_dim, node_dim_list, edge_drop, feat_drop, attention_drop, tau, lam, lam_out,
                 heads, training, lam_out_2):
        # edge_drop边丢弃（随机扰动）, attn_drop 特征丢弃,
        # 原路径模型层数， 近邻模型层数, 隐藏维度, 节点初始维度, 节点丢弃率，注意力丢弃率, 节点类型数量
        super(overall_model, self).__init__()
        self.fc_list = nn.ModuleList([nn.Linear(node_dim, hidden_dim, bias=True)
                                      for node_dim in node_dim_list])  # 节点线性变换
        self.lam = lam
        self.lam_out = lam_out
        self.lam_out_2 = lam_out_2
        # 线性变换层初始化
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)

        # self.meta = meta_path_model(homo_meta_gene_num, homo_meta_dis_num, heter_meta_num, meta_layer_num,
        #                             hidden_dim, hidden_dim, attention_drop, edge_drop, feat_drop)
        self.meta = meta_path_model_2(homo_meta_gene_num, homo_meta_dis_num, meta_layer_num,
                                      hidden_dim, hidden_dim, attention_drop, edge_drop, feat_drop, tau, lam)
        '''
        def forward(self, homo_meta_gene_adj_list, homo_meta_dis_adj_list, heter_meta_path_adj_list, features, training):
        '''
        # self.neigh = neigh_path_model(neigh_layer_num, heads, hidden_dim, hidden_dim, feat_drop, attention_drop, tau,
        #                               lam)

        self.gene_contrast = Contrast(hidden_dim, tau, lam)
        self.disease_contrast = Contrast(hidden_dim, tau, lam)
        self.prediction = MLP(6 * hidden_dim, 2)
        # self.mlp = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        # # self.w = nn.Parameter(nn.Parameter(torch.FloatTensor(hidden_dim * 2, hidden_dim * 2)))
        # # nn.init.xavier_normal_(self.w, gain=1.414)
        # # self.sam_GCN1 = GCN(hidden_dim, hidden_dim)
        # # self.sam_GCN2 = GCN(hidden_dim, hidden_dim)

    def forward(self, features_list, meta_adj_dict, pos_inter, pos_outer, train_target_gene_index,
                train_target_disease_index, train_labels):
        h = []
        for i in range(len(features_list)):
            h.append(F.elu(self.feat_drop(self.fc_list[i](features_list[i]))))
        # meta_gene_features, meta_disease_features = self.meta(meta_adj_dict['g'], meta_adj_dict['d'],
        #                                                       meta_adj_dict['all'], h, training)
        # 带多尺度间损失的
        # meta_gene_features, meta_disease_features, beta, inter_loss, max_loss, loss_outer = \
        #     self.meta(meta_adj_dict['g'], meta_adj_dict['d'], h, pos_inter, pos_outer, training)
        # 不带多尺度间损失的
        # meta_gene_features, meta_disease_features, beta, inter_loss, max_loss = \
        #     self.meta(meta_adj_dict['g'], meta_adj_dict['d'], h, pos_inter, pos_outer, training)
        meta_gene_features, meta_disease_features, beta, inter_loss, max_loss, s_loss = \
            self.meta(meta_adj_dict['g'], meta_adj_dict['d'], h, pos_inter, pos_outer)
        '''
        self, adj, features, training
        '''
        # gene_features = self.sam_GCN1(meta_gene_features, neigh_adj_list[0][0])
        # disease_features = self.sam_GCN2(meta_disease_features, neigh_adj_list[2][2])
        # neigh_gene_feature, neigh_disease_feature, inter_loss = self.neigh(neigh_adj_list, h, pos_inter, training)

        # outer_gene_loss = self.gene_contrast(neigh_gene_feature, meta_gene_features, pos_inter[0])
        # outer_disease_loss = self.disease_contrast(neigh_disease_feature, meta_disease_features, pos_inter[1])
        # outer_loss = (outer_gene_loss + outer_disease_loss) / 2
        # total_loss = self.lam * (outer_gene_loss + outer_disease_loss) + (1 - self.lam) * inter_loss
        # sample = torch.cat([neigh_gene_feature[train_target_gene_index], meta_gene_features[train_target_gene_index],
        #                     neigh_disease_feature[train_target_disease_index],
        #                     meta_disease_features[train_target_disease_index]], 1)
        # gene_feature = F.elu(self.mlp(torch.cat([neigh_gene_feature, meta_gene_features], 1)))
        # disease_feature = F.elu(self.mlp(torch.cat([neigh_disease_feature, meta_disease_features], 1)))
        sample = torch.cat([meta_gene_features[train_target_gene_index],
                            meta_disease_features[train_target_disease_index]], 1)
        # 特征图构建
        y = self.prediction(sample)
        # y_case = F.softmax(y, 1)
        y = F.log_softmax(y, 1)
        train_loss = F.nll_loss(y, train_labels)
        # total_loss = (1 - self.lam_out) * train_loss + self.lam_out * meta_inter_loss        #
        print('对比损失:{:.4f}'.format(inter_loss))
        print('最大对比损失:{:.4f}'.format(max_loss))
        print('任务损失:{:.4f}'.format(train_loss))
        print('多尺度间损失:{:.4f}'.format(s_loss))
        # total_loss = train_loss + self.lam_out * (inter_loss + 0.1 * max_loss ) + loss_outer
        total_loss = self.lam_out_2 * train_loss + self.lam_out * (inter_loss + 0.1 * max_loss + s_loss)  #  + s_loss
        # total_loss = train_loss + self.lam_out * inter_loss
        # total_loss = self.lam_out * train_loss + outer_loss + \
        #              (0.2 * meta_inter_loss + 0.8 * inter_loss) # (1 - self.lam_out) *
        return y, total_loss, beta, sample


class MLP(nn.Module):
    # 128, 2
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),  # 128   64
            nn.ReLU(inplace=True),
            # nn.Linear(input_size // 4, input_size // 8),  # 64 ,32
            # nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, output_size),  # 32, 2

        )

    def forward(self, x):
        out = self.linear(x)
        return out


class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()  # 64 0.9 0.5
        self.proj = nn.Sequential(  # 线性曾
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau  # 0.9
        self.lam = lam  # 0.5
        for model in self.proj:  # 初始化线形层
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)  # 4057，1
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)  # 4057，1
        dot_numerator = torch.mm(z1, z2.t())  # 4057，4057
        dot_denominator = torch.mm(z1_norm, z2_norm.t())  # 4057，4057
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)  # dot_numerator/dot_denominator/0.9
        return sim_matrix

    def forward(self, z_mp, z_sc, pos):
        z_proj_mp = self.proj(z_mp)  # 4057，64
        z_proj_sc = self.proj(z_sc)  # 4057，64
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)  # 4057，4057
        matrix_sc2mp = matrix_mp2sc.t()  # 4057，4057matrix_mp2sc的转置

        matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)  # 4057，4057
        lori_mp = -torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()  # 一个tensor值

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)  # 4057，4057
        lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()  # tensor
        return (lori_mp + lori_sc) / 2
