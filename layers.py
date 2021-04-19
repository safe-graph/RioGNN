import sys
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable


from operator import itemgetter
import math
import numpy as np

"""
	Rio-GNN Layers
	Paper: Reinforced Neighborhood Selection Guided Multi-Relational Graph Neural Networks
	Source: https://github.com/safe-graph/RioGNN
"""


class InterAgg(nn.Module):

	def __init__(self, actor, critic, init_rl, init_action, auc, stop_num, width,
				 features, feature_dim,
				 embed_dim, adj_lists, intraggs,
				 inter, step_size, cuda=True):
		"""
		Initialize the inter-relation aggregator
		:param actor: actors in each depths at each layers for each relation
		:param critic: critics in each depths at each layers for each relation
		:param init_rl:
		:param init_action:
		:param auc:

		:param features: the input node features or embeddings for all nodes
		:param feature_dim: the input dimension
		:param embed_dim: the output dimension
		:param adj_lists: a list of adjacency lists for each single-relation graph
		:param intraggs: the intra-relation aggregators used by each single-relation graph
		:param inter: the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
		:param step_size: the RL action step size
		:param cuda: whether to use GPU
		"""
		super(InterAgg, self).__init__()

		self.features = features
		self.dropout = 0.6
		self.adj_lists = adj_lists
		self.intra_agg1 = intraggs[0]
		self.intra_agg2 = intraggs[1]
		self.intra_agg3 = intraggs[2]
		self.embed_dim = embed_dim
		self.feat_dim = feature_dim
		self.inter = inter
		self.step_size = step_size
		self.cuda = cuda
		self.intra_agg1.cuda = cuda
		self.intra_agg2.cuda = cuda
		self.intra_agg3.cuda = cuda

		# RL condition flag
		self.RL = True
		self.actor = actor
		self.critic = critic
		self.init_rl = init_rl
		self.init_action = init_action
		self.auc = auc
		self.max_auc = 0
		self.max_thresholds = [0, 0, 0]
		self.init_termination = [0, 0, 0]
		self.stop_num = stop_num
		self.width = width

		# number of batches for current epoch, assigned during training
		self.batch_num = 0

		# initial filtering thresholds
		self.thresholds = [0.5, 0.5, 0.5]

		# the activation function used by attention mechanism
		self.leakyrelu = nn.LeakyReLU(0.2)

		# parameter used to transform node embeddings before inter-relation aggregation
		self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim, self.feat_dim))
		init.xavier_uniform_(self.weight)

		# weight parameter for each relation used by Rio-Weight
		self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim, 3))
		init.xavier_uniform_(self.alpha)

		# parameters used by attention layer
		self.a = nn.Parameter(torch.FloatTensor(2 * self.embed_dim, 1))
		init.xavier_uniform_(self.a)

		# label predictor for similarity measure
		self.label_clf = nn.Linear(self.feat_dim, 2)

		# initialize the parameter logs
		self.weights_log = []
		self.thresholds_log = [self.thresholds]
		self.actions_log = []
		self.relation_score_log = []

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
		unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),
								 set.union(*to_neighs[2], set(nodes)))

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
		r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
		r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
		r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

		# assign label-aware scores to neighbor nodes for each batch node and relation
		r1_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r1_list]
		r2_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r2_list]
		r3_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r3_list]

		# count the number of neighbors kept for aggregation for each batch node and relation
		r1_sample_num_list = [math.ceil(len(neighs) * self.thresholds[0]) for neighs in r1_list]
		r2_sample_num_list = [math.ceil(len(neighs) * self.thresholds[1]) for neighs in r2_list]
		r3_sample_num_list = [math.ceil(len(neighs) * self.thresholds[2]) for neighs in r3_list]

		# intra-aggregation steps for each relation
		# Eq. (8) in the paper
		r1_feats, r1_scores = self.intra_agg1.forward(nodes, r1_list, center_scores, r1_scores, r1_sample_num_list)
		r2_feats, r2_scores = self.intra_agg2.forward(nodes, r2_list, center_scores, r2_scores, r2_sample_num_list)
		r3_feats, r3_scores = self.intra_agg3.forward(nodes, r3_list, center_scores, r3_scores, r3_sample_num_list)

		# concat the intra-aggregated embeddings from each relation
		neigh_feats = torch.cat((r1_feats, r2_feats, r3_feats), dim=0)

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
			combined, attention = att_inter_agg(len(self.adj_lists), self.leakyrelu, self_feats, neigh_feats, self.embed_dim,
												self.weight, self.a, n, self.dropout, self.training, self.cuda)
		elif self.inter == 'Weight':
			# 2) Rio-Weight Inter-relation Aggregator
			combined = weight_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight, self.alpha, n, self.cuda)
			gem_weights = F.softmax(torch.sum(self.alpha, dim=0), dim=0).tolist()
			if train_flag:
				print(f'Weights: {gem_weights}')
		elif self.inter == 'Mean':
			# 3) Rio-Mean Inter-relation Aggregator
			combined = mean_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight, n, self.cuda)
		elif self.inter == 'GNN':
			# 4) Rio-GNN Inter-relation Aggregator
			combined = threshold_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight, self.thresholds, n, self.cuda)

		# the reinforcement learning module
		if self.RL and train_flag:
			relation_scores, rewards, thresholds, stop_flag, actions, init_rl, init_action, max_auc, max_thresholds, init_termination = RLModule(self.actor, self.critic, self.init_rl, self.init_action, self.actions_log, self.auc, self.max_auc, self.max_thresholds, self.thresholds_log, self.init_termination, self.stop_num, self.width, [r1_scores, r2_scores, r3_scores], self.relation_score_log, labels, self.thresholds, self.batch_num, self.step_size)
			self.thresholds = thresholds
			self.RL = stop_flag
			self.relation_score_log.append(relation_scores)
			self.thresholds_log.append(self.thresholds)
			self.actions_log.append(actions)
			self.init_rl = init_rl
			self.init_action = init_action
			self.max_auc = max_auc
			self.max_thresholds = max_thresholds
			self.init_termination = init_termination

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


def RLModule(actor, critic, init_rl, init_action, actions_log, auc, max_auc0, max_thresholds, thresholds_log, init_termination, stop_num, width,
			 scores, scores_log, labels, thresholds, batch_num, step_size):
	"""
	The reinforcement learning module.
	It updates the neighbor filtering threshold for each relation based
	on the average neighbor distances between two consecutive epochs.
	:param actor: actors in each depths for each relation
	:param critic: critics in each depths for each relation
	:param init_rl: the RLT deep labeling for each relation
	:param init_action: action interval of current depth for each relation
	:param actions_log: a list stores actions for each batch
	:param auc: the auc of the previous filter thresholds for each relation
	:param max_auc0: the highest auc in history for each relation
	:param max_thresholds: filter threshold corresponding to the highest auc in history for each relation
	:param thresholds_log: a list stores thresholds for each batch
	:param init_termination:
	:param stop_num:
	:param width:

	:param scores: the neighbor nodes label-aware scores for each relation
	:param scores_log: a list stores the relation average distances for each batch
	:param labels: the batch node labels used to select positive nodes
	:param thresholds: the current neighbor filtering thresholds for each relation
	:param batch_num: numbers batches in an epoch
	:param step_size: the RL action step size
	:return relation_scores: the relation average distances for current batch
	:return rewards: the reward for given thresholds in current epoch
	:return new_thresholds: the new filtering thresholds updated according to the rewards
	:return stop_flag: the RL terminal condition flag

	:return actions:
	:return new_init_rl: the new RLT deep labeling for each relation
	:return new_init_action:
	:return new_max_auc:
	:return new_max_thresholds:
	:return new_init_termination:
	"""

	relation_scores = []
	stop_flag = True
	new_init_rl = init_rl
	new_init_action = init_action
	new_max_auc = max_auc0
	new_max_thresholds = max_thresholds
	new_init_termination = [0, 0, 0]


	# only compute the average neighbor distances for positive nodes
	pos_index = (labels == 1).nonzero().tolist()
	pos_index = [i[0] for i in pos_index]

	# compute average neighbor distances for each relation
	for score in scores:
		pos_scores = itemgetter(*pos_index)(score)
		neigh_count = sum([1 if isinstance(i, float) else len(i) for i in pos_scores])
		pos_sum = [i if isinstance(i, float) else sum(i) for i in pos_scores]
		relation_scores.append(sum(pos_sum) / neigh_count)

	# after completing the first epoch
	if len(scores_log) == batch_num:
		new_init_termination = [i + 1 for i in init_termination]
		# STATE
		current_epoch_scores = [sum(s) / batch_num for s in zip(*scores_log[-batch_num:])]
		print("Average distance：" + ','.join([str(current_epoch_scores[i]) for i, s in enumerate(current_epoch_scores)]))
		states = [np.array([s], float) for i, s in enumerate(current_epoch_scores)]
		# ACTION
		actions = [actor[i][new_init_rl[i]].choose_action(s) for i, s in enumerate(states)]
		new_thresholds = [new_init_action[i] + (r + 1) * pow(1/width[i], new_init_rl[i] + 1) for i, r in enumerate(actions)]
		new_thresholds = [1 if r >= 1 else r for i, r in enumerate(new_thresholds)]
		# REWARD
		rewards = [0, 0, 0]

	# during the epoch
	elif len(scores_log) % batch_num != 0 or len(scores_log) < 2 * batch_num:
		new_init_termination = init_termination
		# do not call RL module within the epoch or within the first two epochs
		rewards = [0, 0, 0]
		actions = [0, 0, 0]
		new_thresholds = thresholds

	# after completing each epoch (except the first epoch)
	else:
		# Backtracking
		previous_thresholds = thresholds_log[-batch_num]
		if auc >= new_max_auc:
			new_max_auc = auc
			new_max_thresholds = previous_thresholds
			print("max_thresholds", new_max_thresholds)

		# STATE
		# update thresholds according to average scores in last epoch
		# Eq.(5) in the paper
		previous_epoch_scores = [sum(s) / batch_num for s in zip(*scores_log[-2 * batch_num:-batch_num])]
		current_epoch_scores = [sum(s) / batch_num for s in zip(*scores_log[-batch_num:])]
		previous_actions = actions_log[-batch_num]
		print("Average distance is：" + ','.join([str(current_epoch_scores[i]) for i, s in enumerate(current_epoch_scores)]))

		# REWARD
		# compute reward for each relation and update the thresholds according to reward
		# Eq. (6) in the paper
		rewards = [s if 0 < thresholds[i] and thresholds[i] <= 1 else -100 for i, s in enumerate(current_epoch_scores)]


		previous_epoch_states = [np.array([s], float) for i, s in enumerate(previous_epoch_scores)]
		states = [np.array([s], float) for i, s in enumerate(current_epoch_scores)]

		# after the smallest continuous epoch
		if len(actions_log) > batch_num * stop_num:
			# RLF termination
			flag = 1
			for r in range(3):
				for s in range(stop_num - 1):
					flag = flag * (1 if actions_log[-1 * (s + 1) * batch_num][r] ==
										actions_log[-1 * (s + 2) * batch_num][r] else 0)
			if flag == 1:
				if len(actor[0]) == new_init_rl[0] + 1 and len(actor[1]) == new_init_rl[1] + 1 and len(actor[2]) == new_init_rl[2] + 1 and \
						init_termination[0] > stop_num and init_termination[1] > stop_num and init_termination[2] > stop_num:
					print("Finished！！！！")
					stop_flag = False

			actions = [0, 0, 0]
			new_thresholds = [0, 0, 0]
			for r_num in range(3):
				f = 1
				for s in range(stop_num - 1):
					f = f * (1 if actions_log[-1 * (s + 1) * batch_num][r_num] == actions_log[-1 * (s + 2) * batch_num][r_num] else 0)
					f = f * (1 if init_termination[r_num] > stop_num else 0)
				# go to the next depth
				if f == 1:
					if len(actor[r_num]) == new_init_rl[r_num] + 1:
						new_init_termination[r_num] = init_termination[r_num]
						# ACTION
						actions[r_num] = previous_actions[r_num]
						new_thresholds[r_num] = new_max_thresholds[r_num]
						print("Relation " + str(r_num+1) + "is Max！！！！")
					else:
						new_init_termination[r_num] = 0
						new_init_rl[r_num] = new_init_rl[r_num] + 1
						new_init_action[r_num] = new_max_thresholds[r_num] - (width[r_num]/2) * pow(1/width[r_num], new_init_rl[r_num] + 1)
						print("Relation " + str(r_num+1) + "change to：" + str(new_init_rl[r_num]+1) + "！！！！")
						# ACTION
						actions[r_num] = actor[r_num][new_init_rl[r_num]].choose_action(states[r_num])
						new_thresholds[r_num] = new_init_action[r_num] + (actions[r_num] + 1) * pow(1/width[r_num], new_init_rl[r_num] + 1)
						if new_thresholds[r_num] >= 1:
							new_thresholds[r_num] = 1
				# keep current depth
				else:
					new_init_termination[r_num] = init_termination[r_num] + 1
					# POLICY
					td_error = critic[r_num][new_init_rl[r_num]].train_Q_network(previous_epoch_states[r_num], rewards[r_num], states[r_num])
					actor[r_num][new_init_rl[r_num]].learn(previous_epoch_states[r_num], previous_actions[r_num], td_error)
					# ACTION
					actions[r_num] = actor[r_num][new_init_rl[r_num]].choose_action(states[r_num])
					new_thresholds[r_num] = new_init_action[r_num] + (actions[r_num] + 1) * pow(1/width[r_num], new_init_rl[r_num] + 1)
					if new_thresholds[r_num] >= 1:
						new_thresholds[r_num] = 1
		# within the minimum continuous epoch
		else:
			new_init_termination = [i + 1 for i in init_termination]
			# POLICY
			td_error = [critic[i][new_init_rl[i]].train_Q_network(previous_epoch_states[i], rewards[i], s) for i, s in
						enumerate(states)]  # gradient = grad[r + gamma * V(s_) - V(s)]
			for i, s in enumerate(previous_epoch_states):  # true_gradient = grad[logPi(s,a) * td_error]
				actor[i][new_init_rl[i]].learn(s, previous_actions[i], td_error[i])
			# ACTION
			actions = [actor[i][new_init_rl[i]].choose_action(s) for i, s in enumerate(states)]
			new_thresholds = [new_init_action[i] + (r + 1) * pow(1/width[i], new_init_rl[i] + 1) for i, r in enumerate(actions)]
			new_thresholds = [1 if r >= 1 else r for i, r in enumerate(new_thresholds)]


		print(f'epoch scores: {current_epoch_scores}')
		print(f'rewards: {rewards}')
		print(f'thresholds: {new_thresholds}')

	return relation_scores, rewards, new_thresholds, stop_flag, \
		   actions, new_init_rl, new_init_action, new_max_auc, new_max_thresholds, new_init_termination


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
	combined = torch.cat((center_h.repeat(3, 1), neigh_h), dim=1)
	e = att_layer(combined.mm(a))
	attention = torch.cat((e[0:n, :], e[n:2 * n, :], e[2 * n:3 * n, :]), dim=1)
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
