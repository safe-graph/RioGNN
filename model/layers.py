import sys
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from operator import itemgetter
import math
from RL.rl_model import *

"""
    Rio-GNN Layers
    Paper: Reinforced Neighborhood Selection Guided Multi-Relational Graph Neural Networks
    Source: https://github.com/safe-graph/RioGNN
"""


class InterAgg(nn.Module):

    def __init__(self, width_rl, height_rl, device, LR, GAMMA, stop_num,
                 features, feature_dim,
                 embed_dim, adj_lists, intra_aggs,
                 inter, cuda=True):
        """
        Initialize the inter-relation aggregator
        :param width_rl: width of each relation tree
        :param height_rl: height of each relation tree
        :param device: "cuda" / "cpu"
        :param LR: Actor learning rate (hyper-parameters of AC)
        :param GAMMA: Actor discount factor (hyper-parameters of AC)
        :param stop_num: deep switching or termination conditions
        :param features: the input node features or embeddings for all nodes
        :param feature_dim: the input dimension
        :param embed_dim: the output dimension
        :param adj_lists: a list of adjacency lists for each single-relation graph
        :param intra_aggs: the intra-relation aggregators used by each single-relation graph
        :param inter: the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
        :param cuda: whether to use GPU
        """
        super(InterAgg, self).__init__()

        self.features = features
        self.dropout = 0.6
        self.adj_lists = adj_lists
        self.intra_aggs = intra_aggs
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.inter = inter
        self.cuda = cuda

        # initial filtering thresholds
        self.thresholds = [0.5 for r in range(len(intra_aggs))]

        # RL condition flag
        self.RL = True
        self.rl_tree = RLForest(width_rl, height_rl, device, LR, GAMMA, stop_num, len(intra_aggs))

        # number of batches for current epoch, assigned during training
        self.batch_num = 0
        self.auc = 0

        # the activation function used by attention mechanism
        self.leakyrelu = nn.LeakyReLU(0.2)

        # parameter used to transform node embeddings before inter-relation aggregation
        self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim, self.feat_dim))
        init.xavier_uniform_(self.weight)

        # weight parameter for each relation used by Rio-Weight
        self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim, len(intra_aggs)))
        init.xavier_uniform_(self.alpha)

        # parameters used by attention layer
        self.a = nn.Parameter(torch.FloatTensor(2 * self.embed_dim, 1))
        init.xavier_uniform_(self.a)

        # label predictor for similarity measure
        self.label_clf = nn.Linear(self.feat_dim, 2)

        # initialize the parameter logs
        self.weights_log = []

    def forward(self, nodes, labels, train_flag=True):
        """
        :param nodes: a list of batch node ids
        :param labels: a list of batch node labels, only used by the RLModule
        :param train_flag: indicates whether in training or testing mode
        :return combined: the embeddings of a batch of input node features
        :return center_scores: the label-aware scores of batch nodes
        """

        # extract 1-hop neighbor ids from adj lists of each single-relation graph
        to_neighs = []
        for adj_list in self.adj_lists:
            to_neighs.append([set(adj_list[int(node)]) for node in nodes])

        # find unique nodes and their neighbors used in current batch
        unique_nodes = set.union(*(set.union(*to_neighs[r]) for r in range(len(self.intra_aggs))), set(nodes))

        # calculate label-aware scores
        if self.cuda:
            batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
        else:
            batch_features = self.features(torch.LongTensor(list(unique_nodes)))
        batch_scores = self.label_clf(batch_features)
        id_mapping = {node_id: index for node_id, index in zip(unique_nodes, range(len(unique_nodes)))}

        # the label-aware scores for current batch of nodes
        center_scores = batch_scores[itemgetter(*nodes)(id_mapping), :]

        # get neighbor node id list for each batch node and relation
        r_list = [[list(to_neigh) for to_neigh in to_neighs[r]] for r in range(len(self.intra_aggs))]

        # assign label-aware scores to neighbor nodes for each batch node and relation
        r_scores = [[batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r_list[r]]
                    for r in range(len(self.intra_aggs))]

        # count the number of neighbors kept for aggregation for each batch node and relation
        r_sample_num_list = [[math.ceil(len(neighs) * self.thresholds[r]) for neighs in r_list[r]]
                             for r in range(len(self.intra_aggs))]

        # intra-aggregation steps for each relation
        # Eq. (8) in the paper
        r_feats, r_scores = tuple(
            zip(*list(self.intra_aggs[r].forward(nodes, r_list[r], center_scores, r_scores[r], r_sample_num_list[r])
                      for r in range(len(self.intra_aggs)))))

        # concat the intra-aggregated embeddings from each relation
        neigh_feats = torch.cat(r_feats, dim=0)

        # get features or embeddings for batch nodes
        if self.cuda and isinstance(nodes, list):
            index = torch.LongTensor(nodes).cuda()
        else:
            index = torch.LongTensor(nodes)
        self_feats = self.features(index)

        # number of nodes in a batch
        n = len(nodes)

        # inter-relation aggregation steps
        # Eq. (9) in the paper
        if self.inter == 'Att':
            # 1) Rio-Att Inter-relation Aggregator
            combined, attention = att_inter_agg(len(self.adj_lists), self.leakyrelu, self_feats, neigh_feats,
                                                self.embed_dim,
                                                self.weight, self.a, n, self.dropout, self.training, self.cuda)
        elif self.inter == 'Weight':
            # 2) Rio-Weight Inter-relation Aggregator
            combined = weight_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight,
                                        self.alpha, n, self.cuda)
            gem_weights = F.softmax(torch.sum(self.alpha, dim=0), dim=0).tolist()
            if train_flag:
                print(f'Weights: {gem_weights}')
        elif self.inter == 'Mean':
            # 3) Rio-Mean Inter-relation Aggregator
            combined = mean_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight, n,
                                      self.cuda)
        elif self.inter == 'GNN':
            # 4) Rio-GNN Inter-relation Aggregator
            combined = threshold_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight,
                                           self.thresholds, n, self.cuda)

        # the reinforcement learning module
        if self.RL and train_flag:
            thresholds, stop_flag = self.rl_tree.get_threshold(list(r_scores), labels, self.thresholds, self.batch_num,
                                                               self.auc)
            self.thresholds = thresholds
            self.RL = stop_flag

        return combined, center_scores


class IntraAgg(nn.Module):

    def __init__(self, features, feat_dim, cuda=False):
        """
        Initialize the intra-relation aggregator
        :param features: the input node features or embeddings for all nodes
        :param feat_dim: the input dimension
        :param cuda: whether to use GPU
        """
        super(IntraAgg, self).__init__()

        self.features = features
        self.cuda = cuda
        self.feat_dim = feat_dim

    def forward(self, nodes, to_neighs_list, batch_scores, neigh_scores, sample_list):
        """
        Code partially from https://github.com/williamleif/graphsage-simple/
        :param nodes: list of nodes in a batch
        :param to_neighs_list: neighbor node id list for each batch node in one relation
        :param batch_scores: the label-aware scores of batch nodes
        :param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
        :param sample_list: the number of neighbors kept for each batch node in one relation
        :return to_feats: the aggregated embeddings of batch nodes neighbors in one relation
        :return samp_scores: the average neighbor distances for each relation after filtering
        """

        # filer neighbors under given relation
        samp_neighs, samp_scores = filter_neighs_ada_threshold(batch_scores, neigh_scores, to_neighs_list, sample_list)

        # find the unique nodes among batch nodes and the filtered neighbors
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        # intra-relation aggregation only with sampled neighbors
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        to_feats = F.relu(to_feats)
        return to_feats, samp_scores


def filter_neighs_ada_threshold(center_scores, neigh_scores, neighs_list, sample_list):
    """
    Filter neighbors according label predictor result with adaptive thresholds
    :param center_scores: the label-aware scores of batch nodes
    :param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
    :param neighs_list: neighbor node id list for each batch node in one relation
    :param sample_list: the number of neighbors kept for each batch node in one relation
    :return samp_neighs: the neighbor indices and neighbor simi scores
    :return samp_scores: the average neighbor distances for each relation after filtering
    """

    samp_neighs = []
    samp_scores = []
    for idx, center_score in enumerate(center_scores):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        center_score = center_score.repeat(neigh_score.size()[0], 1)
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]

        # compute the L1-distance of batch nodes and their neighbors
        # Eq. (2) in paper
        score_diff = torch.abs(center_score - neigh_score).squeeze()
        sorted_scores, sorted_indices = torch.sort(score_diff, dim=0, descending=False)
        selected_indices = sorted_indices.tolist()

        # top-p sampling according to distance ranking and thresholds
        # Section 3.3.1 in paper
        if len(neigh_scores[idx]) > num_sample + 1:
            selected_neighs = [neighs_indices[n] for n in selected_indices[:num_sample]]
            selected_scores = sorted_scores.tolist()[:num_sample]
        else:
            selected_neighs = neighs_indices
            selected_scores = score_diff.tolist()
            if isinstance(selected_scores, float):
                selected_scores = [selected_scores]

        samp_neighs.append(set(selected_neighs))
        samp_scores.append(selected_scores)

    return samp_neighs, samp_scores


def mean_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, n, cuda):
    """
    Mean inter-relation aggregator
    :param num_relations: number of relations in the graph
    :param self_feats: batch nodes features or embeddings
    :param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
    :param embed_dim: the dimension of output embedding
    :param weight: parameter used to transform node embeddings before inter-relation aggregation
    :param n: number of nodes in a batch
    :param cuda: whether use GPU
    :return: inter-relation aggregated node embeddings
    """

    # transform batch node embedding and neighbor embedding in each relation with weight parameter
    center_h = weight.mm(self_feats.t())
    neigh_h = weight.mm(neigh_feats.t())

    # initialize the final neighbor embedding
    if cuda:
        aggregated = torch.zeros(size=(embed_dim, n)).cuda()
    else:
        aggregated = torch.zeros(size=(embed_dim, n))

    # sum neighbor embeddings together
    for r in range(num_relations):
        aggregated += neigh_h[:, r * n:(r + 1) * n]

    # sum aggregated neighbor embedding and batch node embedding
    # take the average of embedding and feed them to activation function
    combined = F.relu((center_h + aggregated) / 4.0)

    return combined


def weight_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, alpha, n, cuda):
    """
    Weight inter-relation aggregator
    Reference: https://arxiv.org/abs/2002.12307
    :param num_relations: number of relations in the graph
    :param self_feats: batch nodes features or embeddings
    :param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
    :param embed_dim: the dimension of output embedding
    :param weight: parameter used to transform node embeddings before inter-relation aggregation
    :param alpha: weight parameter for each relation used by Rio-Weight
    :param n: number of nodes in a batch
    :param cuda: whether use GPU
    :return: inter-relation aggregated node embeddings
    """

    # transform batch node embedding and neighbor embedding in each relation with weight parameter
    center_h = weight.mm(self_feats.t())
    neigh_h = weight.mm(neigh_feats.t())

    # compute relation weights using softmax
    w = F.softmax(alpha, dim=1)

    # initialize the final neighbor embedding
    if cuda:
        aggregated = torch.zeros(size=(embed_dim, n)).cuda()
    else:
        aggregated = torch.zeros(size=(embed_dim, n))

    # add weighted neighbor embeddings in each relation together
    for r in range(num_relations):
        aggregated += torch.mul(w[:, r].unsqueeze(1).repeat(1, n), neigh_h[:, r * n:(r + 1) * n])

    # sum aggregated neighbor embedding and batch node embedding
    # feed them to activation function
    combined = F.relu(center_h + aggregated)

    return combined


def att_inter_agg(num_relations, att_layer, self_feats, neigh_feats, embed_dim, weight, a, n, dropout, training, cuda):
    """
    Attention-based inter-relation aggregator
    Reference: https://github.com/Diego999/pyGAT
    :param num_relations: num_relations: number of relations in the graph
    :param att_layer: the activation function used by the attention layer
    :param self_feats: batch nodes features or embeddings
    :param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
    :param embed_dim: the dimension of output embedding
    :param weight: parameter used to transform node embeddings before inter-relation aggregation
    :param a: parameters used by attention layer
    :param n: number of nodes in a batch
    :param dropout: dropout for attention layer
    :param training: a flag indicating whether in the training or testing mode
    :param cuda: whether use GPU
    :return combined: inter-relation aggregated node embeddings
    :return att: the attention weights for each relation
    """

    # transform batch node embedding and neighbor embedding in each relation with weight parameter
    center_h = self_feats.mm(weight.t())
    neigh_h = neigh_feats.mm(weight.t())

    # compute attention weights
    combined = torch.cat((center_h.repeat(num_relations, 1), neigh_h), dim=1)
    e = att_layer(combined.mm(a))
    attention = torch.cat((e[0:n, :], e[n:2 * n, :], e[2 * n:num_relations * n, :]), dim=1)
    ori_attention = F.softmax(attention, dim=1)
    attention = F.dropout(ori_attention, dropout, training=training)

    # initialize the final neighbor embedding
    if cuda:
        aggregated = torch.zeros(size=(n, embed_dim)).cuda()
    else:
        aggregated = torch.zeros(size=(n, embed_dim))

    # add neighbor embeddings in each relation together with attention weights
    for r in range(num_relations):
        aggregated += torch.mul(attention[:, r].unsqueeze(1).repeat(1, embed_dim), neigh_h[r * n:(r + 1) * n, :])

    # sum aggregated neighbor embedding and batch node embedding
    # feed them to activation function
    combined = F.relu((center_h + aggregated).t())

    # extract the attention weights
    att = F.softmax(torch.sum(ori_attention, dim=0), dim=0)

    return combined, att


def threshold_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, threshold, n, cuda):
    """
    Rio-GNN inter-relation aggregator
    Eq. (9) in the paper
    :param num_relations: number of relations in the graph
    :param self_feats: batch nodes features or embeddings
    :param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
    :param embed_dim: the dimension of output embedding
    :param weight: parameter used to transform node embeddings before inter-relation aggregation
    :param threshold: the neighbor filtering thresholds used as aggregating weights
    :param n: number of nodes in a batch
    :param cuda: whether use GPU
    :return: inter-relation aggregated node embeddings
    """

    # transform batch node embedding and neighbor embedding in each relation with weight parameter
    center_h = weight.mm(self_feats.t())
    neigh_h = weight.mm(neigh_feats.t())

    if cuda:
        # use thresholds as aggregating weights
        w = torch.FloatTensor(threshold).repeat(weight.size(0), 1).cuda()

        # initialize the final neighbor embedding
        aggregated = torch.zeros(size=(embed_dim, n)).cuda()
    else:
        w = torch.FloatTensor(threshold).repeat(weight.size(0), 1)
        aggregated = torch.zeros(size=(embed_dim, n))

    # add weighted neighbor embeddings in each relation together
    for r in range(num_relations):
        aggregated += torch.mul(w[:, r].unsqueeze(1).repeat(1, n), neigh_h[:, r * n:(r + 1) * n])

    # sum aggregated neighbor embedding and batch node embedding
    # feed them to activation function
    combined = F.relu(center_h + aggregated)

    return combined
