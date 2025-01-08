import math

import torch.nn as nn
import torch.nn.functional as F
import torch


# from neigh_model import *


class meta_path_model_2(nn.Module):
    def __init__(self, homo_meta_gene_num, homo_meta_dis_num, layer_num, in_dim,
                 hidden_dim, attention_drop, edge_drop, feat_drop, tau, lam, type_num=2):
        super(meta_path_model_2, self).__init__()
        self.gene_meta_num = homo_meta_gene_num
        self.dis_meta_num = homo_meta_dis_num
        self.homo_gene_layer = nn.ModuleList([homo_meta_path_layer_min(layer_num, hidden_dim, hidden_dim,
                                                                       feat_drop, edge_drop, tau, lam)
                                              for _ in range(homo_meta_gene_num * 2)])  # 一个元路径一个层
        self.homo_dis_layer = nn.ModuleList([homo_meta_path_layer_min(layer_num, hidden_dim, hidden_dim,
                                                                      attention_drop, edge_drop, tau, lam)
                                             for _ in range(homo_meta_dis_num * 2)])  # 一个元路径一个层
        self.meta_ave_layer = nn.ModuleList([meta_ave_att(hidden_dim, hidden_dim) for _ in range(type_num)])
        # self.contrast_layer_gene = Max_Contrast(hidden_dim, tau, lam)
        # self.contrast_layer_dis = Max_Contrast(hidden_dim, tau, lam)
        self.contrast_layer_gene = Min_Contrast(hidden_dim, tau, lam)
        self.contrast_layer_dis = Min_Contrast(hidden_dim, tau, lam)
        # self.att = nn.ModuleList(
        #     [Heco_SelfAttention_att(hidden_dim, hidden_dim, attention_drop) for _ in range(type_num)])
        self.att = nn.ModuleList(
            [HAN_SemanticAttention(hidden_dim, hidden_dim) for _ in range(type_num)])
        self.att_sub_gene = nn.ModuleList(
            [HAN_SemanticAttention(hidden_dim, hidden_dim) for _ in range(homo_meta_gene_num)])
        self.att_sub_dis = nn.ModuleList(
            [HAN_SemanticAttention(hidden_dim, hidden_dim) for _ in range(homo_meta_dis_num)])
        self.Lstm = LSTMModel(hidden_dim, hidden_dim, 2, hidden_dim)
        self.tau = tau
        self.lam = lam

    def reset_parameters(self):
        for res in self.att_sub_gene:
            nn.init.xavier_normal_(res.weight, gain=1.414)
        for res in self.att_sub_dis:
            nn.init.xavier_normal_(res.weight, gain=1.414)

    def forward(self, homo_meta_gene_adj_list, homo_meta_dis_adj_list,
                features, pos, pos_outer):
        # 对于基因相关元路径
        homo_meta_gene_feature_list = []
        loss_gene = 0
        max_loss_gene = 0
        s_loss_gene = 0
        all_meta_loss = 0
        gene_c = 0
        for n, item in enumerate(homo_meta_gene_adj_list.items()):
            meta, meta_adj = item
            homo_sub_feature_list = []
            for nn, m in enumerate(meta_adj):
                meta_features, loss_inter, max_loss, struct_loss = self.homo_gene_layer[0](m, features[0],
                                                                                           pos[0],
                                                                                           pos_outer[0])
                gene_c += 1
                # meta_features, loss_inter, max_loss, struct_loss = self.homo_gene_layer[n](m, features[0],
                #                                                                                      pos[0],
                #                                                                                      pos_outer[0],
                #                                                                                      training)
                homo_sub_feature_list.append(meta_features)
                loss_gene += loss_inter
                max_loss_gene += max_loss
                # s_loss_gene += struct_loss
            # gene_output_embedding = self.att_sub_gene[n](torch.cat(homo_sub_feature_list, 1))
            # loss_g1 = self.contrast_layer_gene(homo_sub_feature_list[0], homo_sub_feature_list[1],
            #                                    pos_outer[0])
            # s_loss_gene += loss_g1
            # gene_output_embedding, beta = self.att_sub_gene[n](torch.stack(homo_sub_feature_list, dim=1))
            gene_output_embedding = homo_sub_feature_list[0]
            # gene_output_embedding, beta = self.att_sub_gene[n](torch.stack(homo_sub_feature_list, dim=1))
            # print("分注意力得分:", beta.data.cpu().numpy())
            homo_meta_gene_feature_list.append(gene_output_embedding)

        loss_dis = 0
        max_loss_dis = 0
        s_loss_dis = 0
        homo_meta_dis_feature_list = []
        dis_c = 0
        for n, item in enumerate(homo_meta_dis_adj_list.items()):
            meta, meta_adj = item
            homo_sub_feature_list = []
            for nn, m in enumerate(meta_adj):
                # meta_features, loss_inter, max_loss, struct_loss = self.homo_dis_layer[n * nn + nn](m, features[2],
                #                                                                                     pos[1],
                #                                                                                     pos_outer[1],
                #                                                                                     training)
                meta_features, loss_inter, max_loss, struct_loss = self.homo_dis_layer[0](m, features[2],
                                                                                          pos[1],
                                                                                          pos_outer[1])
                dis_c += 1
                homo_sub_feature_list.append(meta_features)
                loss_dis += loss_inter
                max_loss_dis += max_loss
                # s_loss_dis += struct_loss
            # dis_output_embedding = self.att_sub_dis[n](torch.cat(homo_sub_feature_list, 1))
            # loss_g1 = self.contrast_layer_dis(homo_sub_feature_list[0], homo_sub_feature_list[1],
            #                                   pos_outer[1])
            # s_loss_dis += loss_g1
            dis_output_embedding = homo_sub_feature_list[0]
            # dis_output_embedding, beta = self.att_sub_gene[n](torch.stack(homo_sub_feature_list, dim=1))
            # print("分注意力得分:", beta.data.cpu().numpy())
            homo_meta_dis_feature_list.append(dis_output_embedding)
        # # heter_meta_gene_feature_list = []
        # # heter_meta_disease_feature_list = []
        # # for n, item in enumerate(heter_meta_path_adj_list.items()):
        # #     meta, meta_adj = item
        # #     meta_gene_features, meta_hpo_features = self.heter_layer[n](meta_adj, [features[0],
        # features[2]], training)
        # #     heter_meta_gene_feature_list.append(meta_gene_features)
        # #     heter_meta_disease_feature_list.append(meta_hpo_features)
        #
        # gene_features_list = heter_meta_gene_feature_list + homo_meta_gene_feature_list
        # disease_features_list = heter_meta_disease_feature_list + homo_meta_dis_feature_list
        # for n,i in enumerate(homo_meta_gene_feature_list):
        loss_g1 = self.contrast_layer_gene(homo_meta_gene_feature_list[0], homo_meta_gene_feature_list[1], pos_outer[0])
        loss_g2 = self.contrast_layer_gene(homo_meta_gene_feature_list[0], homo_meta_gene_feature_list[2], pos_outer[0])
        loss_g3 = self.contrast_layer_gene(homo_meta_gene_feature_list[1], homo_meta_gene_feature_list[2], pos_outer[0])
        loss_d1 = self.contrast_layer_dis(homo_meta_dis_feature_list[0], homo_meta_dis_feature_list[1], pos_outer[1])
        loss_d1 = self.contrast_layer_dis(homo_meta_dis_feature_list[0], homo_meta_dis_feature_list[2], pos_outer[1])
        loss_d1 = self.contrast_layer_dis(homo_meta_dis_feature_list[1], homo_meta_dis_feature_list[2], pos_outer[1])
        gene_embeddings = torch.stack(homo_meta_gene_feature_list, dim=1)
        disease_embeddings = torch.stack(homo_meta_dis_feature_list, dim=1)
        # 方法5
        # gene_output = homo_meta_gene_feature_list[0]
        # for kk in homo_meta_gene_feature_list[1:]:
        #     gene_output = gene_output + kk
        # disease_output = homo_meta_dis_feature_list[0]
        # for kk in homo_meta_dis_feature_list[1:]:
        #     disease_output = disease_output + kk
        # gene_output = gene_output/len(homo_meta_gene_feature_list)
        # disease_output = disease_output/len(homo_meta_dis_feature_list)
        # 方法1
        gene_output = torch.cat(homo_meta_gene_feature_list, dim=1)
        disease_output = torch.cat(homo_meta_dis_feature_list, dim=1)
        # 方法2
        # gene_output = homo_meta_gene_feature_list[0]
        # disease_output = homo_meta_dis_feature_list[1]
        # 方法3
        # gene_output, gene_beta = self.att[0](gene_embeddings)
        # disease_output, disease_beta = self.att[-1](disease_embeddings)
        # 方法4
        # gene_output = self.Lstm(gene_embeddings)
        # disease_output = self.Lstm(disease_embeddings)
        # return gene_output, disease_output, [0, 0], \
        #        (loss_gene + loss_dis) / 2, \
        #        (max_loss_gene + max_loss_dis) / 2, (
        #                (loss_g1 + loss_g2 + loss_g3) + (loss_d1 + loss_d2 + loss_d3)) / 2
        return gene_output, disease_output, [0, 0], \
               (loss_gene / (3 * self.gene_meta_num) + loss_dis / (3 * self.dis_meta_num)) / 2, \
               (max_loss_gene / (3 * self.gene_meta_num) + max_loss_dis / (3 * self.dis_meta_num)) / 2, (
                       (loss_g1 + loss_g2 + loss_g3) + (loss_d1)) / 2
        # return gene_output, disease_output, [gene_beta, disease_beta], \
        #        (loss_gene / (3 * self.gene_meta_num) + loss_dis / (3 * self.dis_meta_num)) / 2, \
        #        (max_loss_gene / (3 * self.gene_meta_num) + max_loss_dis / (3 * self.dis_meta_num)) / 2, (
        #                (loss_g1 + loss_g2 + loss_g3) / 3 + (loss_d1 + loss_d2 + loss_d3) / 3) / 2
        # (s_loss_gene / 3 + s_loss_dis / 3) / 2


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        # 根据是否双向，调整全连接层的输入大小
        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        # h0 = torch.zeros(self.num_layers * (2 if self.lstm.bidirectional else 1), x.size(0), self.hidden_size).to(
        #     x.device)
        # c0 = torch.zeros(self.num_layers * (2 if self.lstm.bidirectional else 1), x.size(0), self.hidden_size).to(
        #     x.device)

        # 前向传播 LSTM
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*num_directions)  # , (h0, c0)

        # 解码最后一个时间步的隐藏状态
        out = out[:, -1, :]

        # 全连接层
        out = self.fc(out)

        return out


class HAN_SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(HAN_SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        att = beta
        beta = beta.expand((z.shape[0],) + beta.shape)
        # print("注意力得分:", att.data.cpu().numpy())  # (N, M, 1)

        return (beta * z).sum(1), att


# Heco注意力机制
class Heco_SelfAttention_att(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 attn_drop,
                 ):
        super(Heco_SelfAttention_att, self).__init__()
        self.tanh = nn.Tanh()
        self.type_weight = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(size=(1, out_dim)), requires_grad=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax()
        self.Leak_Relu = nn.LeakyReLU(0.05)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.type_weight.weight, gain=1.414)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

    def forward(self, inputs):
        att_score = []
        attn_curr = self.attn_drop(self.att)
        for i in inputs:
            x = self.Leak_Relu(self.type_weight(i)).mean(dim=0)
            att_score.append(attn_curr.matmul(x.t()))
        att_score = torch.cat(att_score, dim=-1).view(-1)
        beta = self.softmax(att_score)  # 全部平均后注意力机制 而不是每个结点的注意力机制
        print("注意力得分:", beta.data.cpu().numpy())  # semantic attention
        outputs = 0
        for i in range(len(inputs)):
            outputs += inputs[i] * beta[i]
        return F.elu(outputs), beta


def create_augmented_matrix(adj, features):
    augment_features = F.dropout(features, 0.3)
    return augment_features


class homo_meta_path_layer(nn.Module):
    def __init__(self, layer_num, in_dim, hidden_dim, feat_drop, edge_drop, tau, lam):
        super(homo_meta_path_layer, self).__init__()
        self.layer = layer_num
        self.feat_drop = feat_drop
        self.edge_drop = edge_drop
        # self.gc_layer = nn.ModuleList([GraphConvolution(in_dim, hidden_dim) for _ in range(self.layer)])
        self.gc_layer = nn.ModuleList([GCN(in_dim, hidden_dim) for _ in range(self.layer)])
        self.gat_layer = nn.ModuleList([GraphAtteneion(in_dim, hidden_dim, 0.3, True, 0.05) for _ in range(self.layer)])
        self.res_lay = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(self.layer)])
        self.h_mlp = nn.Linear(in_dim * (layer_num + 1), hidden_dim, bias=False)
        self.min_contrast = Min_Contrast(hidden_dim, tau, lam)
        self.l = nn.Linear(in_dim, hidden_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for res in self.res_lay:
            nn.init.xavier_normal_(res.weight, gain=1.414)
        nn.init.xavier_normal_(self.h_mlp.weight, gain=1.414)

    def forward(self, adj, features, pos, pos_outer):
        new_adj = []
        for i in adj:
            new_adj.append(F.dropout(i.to_dense(), self.edge_drop))
        h = []
        h.append([features, features, features])
        pro_x = [features, features, features]
        for lay in range(self.layer):
            # x = self.gc_layer[lay](h[-1], adj)
            x_0 = self.gat_layer[lay](new_adj[0], h[lay][0])
            x_1 = self.gat_layer[lay](new_adj[1], h[lay][1])
            x_2 = self.gat_layer[lay](new_adj[2], h[lay][2])
            # x = F.dropout(x, self.feat_drop, training=training)
            x_0 = x_0 + 0.2 * self.res_lay[lay](pro_x[0])
            x_1 = x_1 + 0.2 * self.res_lay[lay](pro_x[1])
            x_1 = x_1 + 0.2 * self.res_lay[lay](pro_x[2])
            h.append([x_0, x_1, x_2])
            pro_x = [x_0, x_1, x_2]
        result = []
        for k in range(len(adj)):
            result.append(F.elu(self.h_mlp(torch.cat([h[0][k], h[1][k], h[2][k]], 1))))
        loss_1 = self.min_contrast(result[0], result[1], pos)
        loss_2 = self.min_contrast(result[0], result[2], pos)
        return result[0], 0.5 * loss_1 + 0.5 * loss_2


# 最大最小化这篇文章
class homo_meta_path_layer_min(nn.Module):
    def __init__(self, layer_num, in_dim, hidden_dim, feat_drop, edge_drop, tau, lam):
        super(homo_meta_path_layer_min, self).__init__()
        self.layer = layer_num
        self.feat_drop = feat_drop
        self.edge_drop = edge_drop
        # self.gc_layer = nn.ModuleList([GraphConvolution(in_dim, hidden_dim) for _ in range(self.layer)])
        self.gc_layer = nn.ModuleList([GCN(in_dim, hidden_dim) for _ in range(self.layer)])
        self.gat_layer = nn.ModuleList(
            [GraphAtteneion(in_dim, hidden_dim, 0.3, 0.05) for _ in range(self.layer)])
        self.linear_layer = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(self.layer)])
        self.res_lay = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(self.layer)])
        self.h_mlp = nn.Linear(in_dim * (layer_num + 1), hidden_dim, bias=False)
        self.feature_create_graph = loss_create_graph(hidden_dim)
        self.min_contrast = Min_Contrast(hidden_dim, tau, lam)
        self.max_contrast = Max_Contrast(hidden_dim, tau, lam)
        self.l = nn.Linear(in_dim, hidden_dim, bias=False)
        self.att = HAN_SemanticAttention(hidden_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for res in self.res_lay:
            nn.init.xavier_normal_(res.weight, gain=1.414)
        for lin in self.linear_layer:
            nn.init.xavier_normal_(lin.weight, gain=1.414)
        nn.init.xavier_normal_(self.h_mlp.weight, gain=1.414)

    def forward(self, adj, features, pos, pos_outer):
        features_aug = features + F.normalize(torch.normal(0, torch.ones_like(features) * 0.01), dim=1)
        new_adj = F.dropout(adj.to_dense(), self.edge_drop)
        h = []
        h.append([features, features_aug, features])
        pro_x = [features, features_aug, features]
        for lay in range(self.layer):
            # x = self.gc_layer[lay](h[-1], adj)
            x_0 = self.gat_layer[lay](new_adj, h[lay][0])
            x_1 = self.gat_layer[lay](new_adj, h[lay][1])
            x_2 = F.relu(self.linear_layer[lay](h[lay][2]))
            # x = F.dropout(x, self.feat_drop, training=training)
            x_0 = x_0 + 0.2 * self.res_lay[lay](pro_x[0])
            x_1 = x_1 + 0.2 * self.res_lay[lay](pro_x[1])
            x_1 = x_1 + 0.2 * self.res_lay[lay](pro_x[2])
            h.append([x_0, x_1, x_2])
            pro_x = [x_0, x_1, x_2]
        result = []
        for k in range(3):
            # a = []
            # # for kk in range(self.layer + 1):
            # #     a.append(h[kk][k])
            result.append(F.elu(self.h_mlp(torch.cat([h[0][k], h[1][k]], 1))))

        loss_1 = self.min_contrast(result[0], result[1], pos_outer)
        loss_2 = self.min_contrast(result[0], result[2], pos_outer)
        loss_3 = self.max_contrast(result[1], result[2], pos_outer)
        gene_embeddings = torch.stack(result, dim=1)
        gene_output, gene_beta = self.att(gene_embeddings)
        return gene_output, 0.5 * loss_1 + 0.5 * loss_2, loss_3, 0
        # return result[0], 0, 0, 0


class loss_create_graph(nn.Module):
    def __init__(self, in_features):
        super(loss_create_graph, self).__init__()
        self.in_features = in_features
        self.bilinear = nn.Linear(2 * in_features, 2, bias=False)

    def forward(self, x, adj):
        all_loss = 0
        new_adj = torch.where(adj > 0, float(1), float(0)).type(torch.FloatTensor).cuda()
        n = 0
        for i, j in zip(x, new_adj):
            z = x[n].unsqueeze(dim=0)
            l = new_adj[n].unsqueeze(dim=0)  # .repeat(1, x.size()[0])  # * torch.ones_like(x)
            z = z.cuda()  # .repeat(1, x.size[0])
            # z = torch  # .repeat(1, x.size[0])
            y = F.sigmoid(torch.matmul(z, x.t()))
            # print(y.transpose(1, 0).size())
            # print(j.unsqueeze(1).type(torch.LongTensor).size())
            train_loss = F.binary_cross_entropy_with_logits(y.type(torch.FloatTensor),
                                                            l.type(torch.FloatTensor))
            all_loss += train_loss
        all_loss = all_loss / x.size()[0]
        return all_loss


# 注意力机制聚合
class GraphAtteneion(nn.Module):
    def __init__(self, in_features, out_features, attention_drop, bias=True):
        super(GraphAtteneion, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.attention_drop = attention_drop
        self.weight = torch.nn.Linear(in_features, out_features, bias=False)
        self.att_src = nn.Parameter(torch.FloatTensor(1, in_features))
        self.att_dst = nn.Parameter(torch.FloatTensor(1, in_features))
        self.leaky_relu = nn.LeakyReLU(0.05)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.weight, gain=1.414)
        nn.init.xavier_normal_(self.att_src, gain=1.414)
        nn.init.xavier_normal_(self.att_dst, gain=1.414)

    def forward(self, adj, features):
        # att_src = F.dropout(self.att_src, self.attention_drop, training=training)
        # att_dst = F.dropout(self.att_dst, self.attention_drop, training=training)
        # att_src = self.att_src  # F.dropout(self.att_src, self.attention_drop)
        # att_dst = self.att_dst  # F.dropout(self.att_dst, self.attention_drop)      # , training=self.training
        x = F.tanh(self.weight(features))
        src_atten_value = (self.att_src * x).sum(dim=1).unsqueeze(-1).repeat(1, x.size()[0])
        dst_atten_value = (self.att_dst * x).sum(dim=1).unsqueeze(-1) \
            .repeat(1, x.size()[0]).transpose(0, 1)
        atten = self.leaky_relu(src_atten_value + dst_atten_value)
        zero_vec = -9e25 * torch.ones_like(atten)  #
        attention = torch.where(adj > 0, atten, zero_vec)
        atten_softmax = F.softmax(attention, dim=-1)
        output = torch.matmul(atten_softmax, features)
        return output


# 注意力网络
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # Xavier均匀分布初始化
        # xavier初始化方法中服从均匀分布U(−a,a) ，分布的参数a = gain * sqrt(6/fan_in+fan_out)，这里有一个gain，增益的大小是依据激活函数类型来设定

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features) #torch.mm矩阵相乘
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(
            torch.matmul(a_input, self.a).squeeze(2))  # torch.matmul矩阵乘法，输入可以是高维；squeeze可以将维度为1的那个维度去掉，当输入值中
        # 存在dim时，则只有当dim对应的维度为1时会实现降维
        # eij = a([Whi||Whj]),j属于Ni

        zero_vec = -9e15 * torch.ones_like(e)  # 范围维度和e一样的全1的矩阵
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # epeat_interleave()：在原有的tensor上，按每一个tensor复制。
        # repeat()：根据原有的tensor复制n个，然后拼接在一起。
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # dim=0表示按行拼接，1表示按列拼接
        # all_combinations_matrix.shape == (N * N, 2 * out_features)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)
        # torch中的view()的作用相当于numpy中的reshape，重新定义矩阵的形状。

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# 类型注意力
class Type_GraphAtteneion(nn.Module):
    def __init__(self, in_features, out_features, attention_drop, bias=True):
        super(Type_GraphAtteneion, self).__init__()
        self.gene_weight = nn.Linear(in_features, out_features, bias=False)
        self.hpo_weight = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.attention_drop = attention_drop
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.gene_weight.weight, gain=gain)
        nn.init.xavier_normal_(self.hpo_weight.weight, gain=gain)
        nn.init.xavier_normal_(self.a, gain=gain)

    def forward(self, feature1, feature2):
        # print('类型注意力 + 边注意力')
        n_feature1 = F.tanh(self.gene_weight(feature1)).mean(dim=0).unsqueeze(0)  # 4454 * 64
        n_feature2 = F.tanh(self.hpo_weight(feature2)).mean(dim=0).unsqueeze(0)  # 4454 * 64
        feature = torch.cat([n_feature1, n_feature2], dim=0)  # 4454 * 2 * 64
        att = torch.matmul(feature, self.a).transpose(0, 1)  # 1, 2
        att = F.leaky_relu(att)
        att = F.softmax(att, dim=-1)  # 2, 1 -> 2, 64
        att = torch.stack([att] * feature1.size()[0], dim=0)
        outputs = torch.matmul(att, torch.stack([feature1, feature2], dim=1)).squeeze(1)  # 4454 * 2 * 64
        return outputs


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class meta_ave_att(nn.Module):
    def __init__(self, in_features, out_features):
        super(meta_ave_att, self).__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.FloatTensor(in_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.weight.weight, gain=gain)
        nn.init.xavier_normal_(self.a, gain=gain)

    def forward(self, feature):
        # print('类型注意力 + 边注意力')
        n_feature = F.tanh(self.weight(feature)).transpose(0, 1).mean(dim=1)  # xx * 1, 64
        att = torch.matmul(n_feature, self.a).transpose(0, 1)  # 1, 2
        att = F.leaky_relu(att)
        att = F.softmax(att, dim=-1)  # 2, 1 -> 2, 64
        att = torch.stack([att] * feature.size()[0], dim=0)
        outputs = F.elu(torch.matmul(att, feature).squeeze(1))  # 4454 * 2 * 64
        return outputs


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=None):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)  # 64 *64 线性层
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)  # 特征线性变换
        out = torch.spmm(adj, seq_fts)  # 元路径聚合
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class Max_Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Max_Contrast, self).__init__()  # 64 0.9 0.5
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
        lori_mp = torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()  # 一个tensor值

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)  # 4057，4057
        lori_sc = torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()  # tensor
        return (lori_mp + lori_sc) / 2
        # return lori_mp


class Min_Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Min_Contrast, self).__init__()  # 64 0.9 0.5
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
        return self.lam * lori_mp + (1 - self.lam) * lori_sc
        # return lori_mp
