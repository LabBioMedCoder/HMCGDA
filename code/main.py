import argparse
import time

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.metrics import auc as auc3
import pandas as pd
# from utils_dataset2 import *
from utils import *
from model import *


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def train(args):
    cudaMsg = torch.cuda.is_available()
    gpuCount = torch.cuda.device_count()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        print('使用GPU加速')
        print('内部损失比例:{}'.format(args.lam_out_2))
        print('损失比例:{}'.format(args.lam_out))
        print('系数:{}'.format(args.lam_out))
        print('学习率:{}'.format(args.lr))
    else:
        device = torch.device("cpu")
        print('使用GPU加速')
    print('邻居{}'.format(args.ma))
    total_result = []
    for r in [1]:  # , 2, 3, 4, 5
        features_set, meta_path_set, train_sample_list, test_sample_list, pos_inter, pos_outer \
            = load_data(r, args)
        print('数据加载完毕！！！')
        if torch.cuda.is_available():
            features_list = [mat2tensor(features).cuda() for features in features_set]
            for n, i in enumerate(pos_inter):
                pos_inter[n] = i.cuda()
            for n, i in enumerate(pos_outer):
                pos_outer[n] = i.cuda()
            for i, j in meta_path_set.items():
                for k, e in j.items():
                    for n, q in enumerate(e):
                        meta_path_set[i][k][n] = q.cuda()

            # 下游任务正负样本
            train_labels_gpu = torch.LongTensor(train_sample_list[-1]).cuda()  # 标签【1，4025】
            test_labels_gpu = torch.LongTensor(test_sample_list[-1]).cuda()
            train_labels = torch.LongTensor(train_sample_list[-1])  # 标签【1，4025】
            test_labels = torch.LongTensor(test_sample_list[-1])
            train_target_gene_index = torch.LongTensor(train_sample_list[0]).cuda()
            train_target_disease_index = torch.LongTensor(train_sample_list[1]).cuda()
            test_target_gene_index = torch.LongTensor(test_sample_list[0]).cuda()
            test_target_disease_index = torch.LongTensor(test_sample_list[1]).cuda()
            node_dim_list = []
            for i in features_set:
                node_dim_list.append(i.shape[1])
            model = overall_model(len(meta_path_set['g'].keys()), len(meta_path_set['d'].keys()), args.meta_layer_num,
                                  args.neigh_layer_num, args.hidden_dim, node_dim_list, args.edge_drop, args.feat_drop,
                                  args.attention_drop, args.tau, args.lam, args.lam_out, args.heads, True,
                                  args.lam_out_2)
            model.cuda()
        else:
            features_list = [mat2tensor(features) for features in features_set]
            for n, i in enumerate(pos_inter):
                pos_inter[n] = i
            for n, i in enumerate(pos_outer):
                pos_outer[n] = i
            cc = 0
            for i, j in meta_path_set.items():
                for k, e in j.items():
                    meta_path_set[i][k] = e
                    cc += 1
            print(cc)
            # edges_self = []
            # for j in [4454, 8092, 3567]:
            #     row = [i for i in range(j)]
            #     col = [i for i in range(j)]
            #     value = [1 for i in range(j)]
            #     adjM_relation = sp.coo_matrix((value, (row, col)), shape=[j, j])
            #     edges_self.append(feature_process(adjM_relation))
            # 构建近邻矩阵
            # 下游任务正负样本
            train_labels_gpu = torch.LongTensor(train_sample_list[-1])  # 标签【1，4025】
            test_labels_gpu = torch.LongTensor(test_sample_list[-1])
            train_labels = torch.LongTensor(train_sample_list[-1])  # 标签【1，4025】
            test_labels = torch.LongTensor(test_sample_list[-1])
            c = len(train_labels)
            train_target_gene_index = torch.LongTensor(train_sample_list[0])
            train_target_disease_index = torch.LongTensor(train_sample_list[1])
            test_target_gene_index = torch.LongTensor(test_sample_list[0])
            test_target_disease_index = torch.LongTensor(test_sample_list[1])
            print(1)
            node_dim_list = []
            for i in features_set:
                node_dim_list.append(i.shape[1])
            # homo_meta_gene_num, homo_meta_dis_num, heter_meta_num, meta_layer_num,  # 元路径那块参数
            # neigh_layer_num, hidden_dim, node_dim_list, edge_drop, feat_drop, training):
            model = overall_model(len(meta_path_set['g'].keys()), len(meta_path_set['d'].keys()), args.meta_layer_num,
                                  args.neigh_layer_num, args.hidden_dim, node_dim_list, args.edge_drop, args.feat_drop,
                                  args.attention_drop, args.tau, args.lam, args.lam_out, args.heads, True,
                                  args.lam_out_2)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True,  # 30
                                       save_path='checkpoint/checkpoint_{}.pt'.format(r))
        best_result = 0
        best_auc = 0
        best_aupr = 0
        best_f1 = 0
        best_new_f1 = 0
        best_accuracy = 0
        best_precession = 0
        best_recall = 0
        best_y = 0
        best_beta = 0
        for epoch in range(args.epoch):
            # print('epoch')
            t_start = time.time()
            # training
            model.train()
            y, train_loss, _, _ = model(features_list, meta_path_set, pos_inter, pos_outer,
                                        train_target_gene_index, train_target_disease_index, train_labels_gpu)
            y = y.cpu()
            train_loss = train_loss.cpu()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # for name, parms in model.named_parameters():
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_requirs:', parms.requires_grad)
            #     print('-->grad_value:', parms.grad)
            #     print("===")
            # time.sleep(60)
            t_end = time.time()
            train_acc = (y.argmax(dim=1) == train_labels).sum().type(
                torch.float) / float(len(train_labels))  # 准确率 y.argmax是求横行最大值索引非0及1
            train_auc = roc_auc_score(train_labels.detach().numpy(), y[:, 1].detach().numpy())
            train_precision, train_recall, thresholds = precision_recall_curve(train_labels.detach().numpy(),
                                                                               y[:, 1].detach().numpy())
            d2 = precision_score(train_labels.detach().numpy(), y.argmax(dim=1).detach().numpy())
            d1 = recall_score(train_labels.detach().numpy(), y.argmax(dim=1).detach().numpy())
            train_aupr = auc3(train_recall, train_precision)
            train_f1 = f1_score(train_labels, y.argmax(dim=1))
            theshold = np.linspace(np.min(y[:, 1].detach().numpy()), np.max(y[:, 1].detach().numpy()), 100)
            train_f1_new = np.max([f1_score(train_labels, y[:, 1].detach().numpy() >= theta) for theta in theshold])
            print('Fold  {} | Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'
                  .format(r, epoch, train_loss.item(), t_end - t_start))
            print("train_auc= {:.4f}".format(train_auc.item()),
                  "train_aupr= {:.4f}".format(train_aupr.item()),
                  "train_accuracy= {:.4f}".format(train_acc.item()),
                  "train_f1= {:.4f}".format(train_f1.item()),
                  "train_new_f1= {:.4f}".format(train_f1_new.item()),
                  "train_precession={:.4f}".format(d2.item()),
                  "train_recall={:.4f}".format(d1.item()),
                  )
            with torch.no_grad():
                model.eval()
                y, val_loss, beta, _ = model(features_list, meta_path_set, pos_inter, pos_outer,
                                             test_target_gene_index, test_target_disease_index, test_labels_gpu)
                y = y.cpu()
                val_loss = val_loss.cpu()

                t_end1 = time.time()
                train_acc = (y.argmax(dim=1) == test_labels).sum().type(
                    torch.float) / float(len(test_labels))  # 准确率 y.argmax是求横行最大值索引非0及1
                train_auc = roc_auc_score(test_labels.detach().numpy(), y[:, 1].detach().numpy())
                train_precision, train_recall, thresholds = precision_recall_curve(test_labels.detach().numpy(),
                                                                                   y[:, 1].detach().numpy())
                d2 = precision_score(test_labels.detach().numpy(), y.argmax(dim=1).detach().numpy())
                d1 = recall_score(test_labels.detach().numpy(), y.argmax(dim=1).detach().numpy())
                train_aupr = auc3(train_recall, train_precision)
                train_f1 = f1_score(test_labels, y.argmax(dim=1))
                theshold = np.linspace(np.min(y[:, 1].detach().numpy()), np.max(y[:, 1].detach().numpy()), 100)

                train_f1_new = np.max(
                    [f1_score(test_labels, y[:, 1].detach().numpy() >= theta) for theta in theshold])
                print('Fold  {} | Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                    r, epoch, val_loss.item(), t_end - t_start))
                print("val_auc= {:.4f}".format(train_auc.item()),
                      "val_aupr= {:.4f}".format(train_aupr.item()),
                      "val_accuracy= {:.4f}".format(train_acc.item()),
                      "val_f1= {:.4f}".format(train_f1.item()),
                      "val_new_f1= {:.4f}".format(train_f1_new.item()),
                      "val_precession={:.4f}".format(d2.item()),
                      "val_recall={:.4f}".format(d1.item()),
                      )
                if (train_auc + train_f1 + train_aupr + train_acc) / 4 > (
                        best_aupr + best_auc + best_f1 + best_accuracy) / 4:
                    best_aupr = train_aupr
                    best_auc = train_auc
                    best_f1 = train_f1
                    best_new_f1 = train_f1_new
                    best_precession = d2
                    best_recall = d1
                    best_accuracy = train_acc
                print("Best test results:",
                      "test_auc= {:.4f}".format(best_auc),
                      "test_aupr= {:.4f}".format(best_aupr),
                      "test_accuracy= {:.4f}".format(best_accuracy),
                      "test_f1= {:.4f}".format(best_f1),
                      "test_precession= {:.4f}".format(best_precession),
                      "test_recall= {:.4f}".format(best_recall),
                      )
                # # best_result = logp.detach().numpy()
                # torch.save(model.state_dict(), 'model_p/ours_param_{}_{}.pth'.format(args.lam_out, r))
                # torch.save(model, 'model_p/ours_model_{}_{}.pth'.format(args.lam_out, r))
                # print('模型保存完毕!!!!')
                early_stopping(train_loss, model)
                if early_stopping.early_stop:
                    print('Early stopping!')
                    total_result.append(
                        [best_auc, best_aupr, best_f1, best_accuracy.item(), best_precession,
                         best_recall])
                    with open('best_ours_embeding_sim_view_{}_struct.plk'.format(r), 'wb') as fe:
                        pickle.dump(best_y, fe)
                    # np.save('y_case.npy', best_y)
                    #     json.dump([best_auc, best_aupr, best_f1, best_accuracy.item(), best_precession,
                    #                best_recall, best_new_f1], f69, indent=2)
                    break
                if epoch == 999:
                    total_result.append(
                        [best_auc, best_aupr, best_f1, best_accuracy.item(), best_precession, best_recall])
                    with open('best_ours_embeding_sim_view_{}_struct.plk'.format(r), 'wb') as fe:
                        pickle.dump(best_y, fe)
                    # np.save('y_case.npy', best_y)
                    # with open('y_result/y_{}_{}.json'.format(args.lam_out, r), 'w') as f69:
                    #     json.dump(best_y, f69, indent=2)
    print(total_result)
    # print(best_beta[0].data.detach().cpu().numpy())
    # print(best_beta[1].data.detach().cpu().numpy())
    # with open('result/result_50pos.json', 'w') as f69:
    #     # with open('../pro_result/r(no_logic_att_result).json', 'w') as f69:
    #     json.dump(total_result, f69, indent=2)
    evaluate(total_result)


def evaluate(total_result):
    best_data_list = []
    auc = 0
    aupr = 0
    f1 = 0
    accuracy = 0
    precession = 0
    recall = 0
    for z in total_result:
        auc += z[0]
        aupr += z[1]
        f1 += z[2]
        accuracy += z[3]
        precession += z[4]
        recall += z[5]
        score = 0
        for zz in z[0:4]:
            score += zz
        best_data_list.append(score / 4)
    max_index = best_data_list.index(max(best_data_list))
    min_index = best_data_list.index(min(best_data_list))
    max_r = total_result[max_index]
    min_r = total_result[min_index]
    mid_r = [auc / 5, aupr / 5, f1 / 5, accuracy / 5, precession / 5, recall / 5]
    s_max = np.var(total_result, axis=0)
    s_norm = np.std(total_result, axis=0)
    print("最好结果:",
          "best_auc= {:.4f}".format(max_r[0]),
          "best_aupr= {:.4f}".format(max_r[1]),
          "best_f1= {:.4f}".format(max_r[2]),
          "best_accuracy= {:.4f}".format(max_r[3]),
          "best_precession={:.4f}".format(max_r[4]),
          "best_recall={:.4f}".format(max_r[5]),
          )
    print("最差结果:",
          "best_auc= {:.4f}".format(min_r[0]),
          "best_aupr= {:.4f}".format(min_r[1]),
          "best_f1= {:.4f}".format(min_r[2]),
          "best_accuracy= {:.4f}".format(min_r[3]),
          "best_precession={:.4f}".format(min_r[4]),
          "best_recall={:.4f}".format(min_r[5]),
          )
    print("平均结果:",
          "best_auc= {:.4f}".format(mid_r[0]),
          "best_aupr= {:.4f}".format(mid_r[1]),
          "best_f1= {:.4f}".format(mid_r[2]),
          "best_accuracy= {:.4f}".format(mid_r[3]),
          "best_precession={:.4f}".format(mid_r[4]),
          "best_recall={:.4f}".format(mid_r[5]),
          )
    print("方差:",
          "best_auc= {:.4f}".format(s_max[0]),
          "best_aupr= {:.4f}".format(s_max[1]),
          "best_f1= {:.4f}".format(s_max[2]),
          "best_accuracy= {:.4f}".format(s_max[3]),
          "best_precession={:.4f}".format(s_max[4]),
          "best_recall={:.4f}".format(s_max[5]),
          )
    print("标准差:",
          "best_auc= {:.4f}".format(s_norm[0]),
          "best_aupr= {:.4f}".format(s_norm[1]),
          "best_f1= {:.4f}".format(s_norm[2]),
          "best_accuracy= {:.4f}".format(s_norm[3]),
          "best_precession={:.4f}".format(s_norm[4]),
          "best_recall={:.4f}".format(s_norm[5]),
          )
    print(total_result)
    total_result.append(max_r)
    total_result.append(min_r)
    total_result.append(mid_r)
    total_result.append(s_max)
    total_result.append(s_norm)
    df = pd.DataFrame(total_result, columns=['AUC', 'AUPR', 'F1', 'ACC', 'Precession', 'recall'],
                      index=['fold1', 'fold2', 'fold3', 'fold4', 'fold5', '最好结果', '最坏结果', '平均结果', '方差', '标准差'])
    print(total_result)
    # df.to_excel('pro_result/result_(lr_{}).xlsx'.format(args.lr))
    df.to_excel('result/result_(dataset2_{}).xlsx'.format(args.ma))  # .format(args.lam_out_2)
    # df.to_excel('../pro_result/result_(no_logic_att_result).xlsx')


def prediction(args):
    net = torch.load('model_p/ours_model_0.3_8.pth')
    net.eval()
    net.cpu()
    features_set, relation_list, meta_path_set, train_sample_list, test_sample_list, pos_inter, pos_outer \
        = load_data(1, args)
    print('数据加载完毕！！！')
    features_list = [mat2tensor(features) for features in features_set]
    for n, i in enumerate(pos_inter):
        pos_inter[n] = i
    for n, i in enumerate(pos_outer):
        pos_outer[n] = i
    cc = 0
    for i, j in meta_path_set.items():
        for k, e in j.items():
            meta_path_set[i][k] = e
            cc += 1
    print(cc)
    # edges_self = []
    # for j in [4454, 8092, 3567]:
    #     row = [i for i in range(j)]
    #     col = [i for i in range(j)]
    #     value = [1 for i in range(j)]
    #     adjM_relation = sp.coo_matrix((value, (row, col)), shape=[j, j])
    #     edges_self.append(feature_process(adjM_relation))
    # 构建近邻矩阵
    neigh_adj_list = [[relation_list['gg'], relation_list['gh'], relation_list['gd']],
                      [relation_list['hg'], relation_list['hh'], relation_list['hd']],
                      [relation_list['dg'], relation_list['dh'], relation_list['dd']]]
    # 下游任务正负样本
    train_labels_gpu = torch.LongTensor(train_sample_list[-1])  # 标签【1，4025】
    test_labels_gpu = torch.LongTensor(test_sample_list[-1])
    train_labels = torch.LongTensor(train_sample_list[-1])  # 标签【1，4025】
    test_labels = torch.LongTensor(test_sample_list[-1])
    c = len(train_labels)
    train_target_gene_index = torch.LongTensor(train_sample_list[0])
    train_target_disease_index = torch.LongTensor(train_sample_list[1])
    test_target_gene_index = torch.LongTensor(test_sample_list[0])
    test_target_disease_index = torch.LongTensor(test_sample_list[1])
    print(1)
    y, val_loss = net(features_list, meta_path_set, neigh_adj_list, pos_inter, pos_outer,
                      test_target_gene_index, test_target_disease_index, test_labels_gpu, False)
    y = y.cpu()
    print(y.shape)
    with open('y_result/y_case_x.json', 'w') as f:
        json.dump(y.detach().numpy().tolist(), f, indent=6)
    # print validation info


if __name__ == '__main__':
    for k in [50]:
        ap = argparse.ArgumentParser(description='gene-disease-hpo数据库')
        ap.add_argument('--top_k', type=int, default=5,
                        help='Type of the node features used.')
        ap.add_argument('--hidden-dim', type=int, default=32, help='节点隐藏状态的维度。默认值为64.')
        ap.add_argument('--heads', type=int, default=4, help='注意头的数量。默认值为8.')
        ap.add_argument('--epoch', type=int, default=1000, help='epochs数量.')
        ap.add_argument('--patience', type=int, default=50, help='Patience.')
        ap.add_argument('--repeat', type=int, default=1, help='重复训练和测试N次。默认值为1.')
        ap.add_argument('--lr', type=float, default=5e-3)  # 5e-3  0.05 0.01 0.005 0.001 0.0005
        ap.add_argument('--weight-decay', type=float, default=5e-4)
        ap.add_argument('--meta-path', type=list, default=[['gh', 'hh', 'hg'], ['dh', 'hh', 'hd']])
        # 边丢弃 特征丢弃 注意力丢弃
        ap.add_argument('--edge-drop', type=float, default=0.3)
        ap.add_argument('--feat-drop', type=float, default=0.3)
        ap.add_argument('--attention_drop', type=float, default=0.3)
        ap.add_argument('--node_type_num', type=int, default=3)
        ap.add_argument('--run', type=int, default=1)
        # 残差比
        ap.add_argument('--res', type=float, default=0.2)
        # 层数
        ap.add_argument('--meta-layer-num', type=int, default=1)
        ap.add_argument('--neigh-layer-num', type=int, default=2)
        ap.add_argument('--tau', type=float, default=0.5)
        ap.add_argument('--lam', type=float, default=0.5)
        ap.add_argument('--lam_out', type=float, default=1)  # 0.7
        ap.add_argument('--lam_out_2', type=float, default=0.4)
        ap.add_argument('--mn', type=int, default=6)
        ap.add_argument('--ma', type=int, default=k)

        args = ap.parse_args()
        train(args)
        # prediction(args)
