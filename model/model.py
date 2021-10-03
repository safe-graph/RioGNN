import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

"""
	Rio-GNN Models
	Paper: Reinforced Neighborhood Selection Guided Multi-Relational Graph Neural Networks
	Source: https://github.com/safe-graph/RioGNN
"""


class OneLayerRio(nn.Module):
	"""
	The Rio-GNN model in one layer
	"""

	def __init__(self, num_classes, inter1, lambda_1):
		"""
		Initialize the Rio-GNN model
		:param num_classes: number of classes (2 in our paper)
		:param inter1: the inter-relation aggregator that output the final embedding
		"""
		super(OneLayerRio, self).__init__()
		self.inter1 = inter1
		self.xent = nn.CrossEntropyLoss()

		# the parameter to transform the final embedding
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, inter1.embed_dim))
		init.xavier_uniform_(self.weight)
		self.lambda_1 = lambda_1

	def forward(self, nodes, labels, train_flag=True):
		embeds1, label_scores = self.inter1(nodes, labels, train_flag)
		scores = self.weight.mm(embeds1)
		return scores.t(), label_scores

	def to_prob(self, nodes, labels, train_flag=True):
		gnn_logits, label_logits = self.forward(nodes, labels, train_flag)
		gnn_scores = torch.sigmoid(gnn_logits)
		label_scores = torch.sigmoid(label_logits)
		return gnn_scores, label_scores

	def loss(self, nodes, labels, train_flag=True):
		gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
		# Simi loss, Eq. (4) in the paper
		label_loss = self.xent(label_scores, labels.squeeze())
		# GNN loss, Eq. (10) in the paper
		gnn_loss = self.xent(gnn_scores, labels.squeeze())
		# the loss function of Rio-GNN, Eq. (11) in the paper
		final_loss = gnn_loss + self.lambda_1 * label_loss
		return final_loss




class TwoLayerRio(nn.Module):
	"""
	The Rio-GNN model in one layer
	"""

	def __init__(self, num_classes, inter1, inter2, lambda_1, last_label_scores):
		"""
		Initialize the Rio-GNN model
		:param num_classes: number of classes (2 in our paper)
		:param inter1: the inter-relation aggregator that output the final embedding
		"""
		super(TwoLayerRio, self).__init__()
		self.inter1 = inter1
		self.inter2 = inter2
		self.xent = nn.CrossEntropyLoss()

		# the parameter to transform the final embedding
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, inter2.embed_dim))
		init.xavier_uniform_(self.weight)
		self.lambda_1 = lambda_1
		self.last_label_scores = last_label_scores

	def forward(self, nodes, labels, train_flag=True):
		label_scores_one = self.last_label_scores
		embeds2, label_scores_two = self.inter2(nodes, labels, train_flag)
		scores2 = self.weight.mm(embeds2)
		return scores2.t(), label_scores_one, label_scores_two

	def to_prob(self, nodes, labels, train_flag=True):
		gnn_logits2, label_logits_one, label_logits_two = self.forward(nodes, labels, train_flag)
		gnn_scores2 = torch.sigmoid(gnn_logits2)
		label_scores_one = torch.sigmoid(label_logits_one)
		label_scores_two = torch.sigmoid(label_logits_two)
		return gnn_scores2, label_scores_one, label_scores_two

	def loss(self, nodes, labels, train_flag=True):
		gnn_scores2, label_scores_one, label_scores_two = self.forward(nodes, labels, train_flag)
		# Simi loss, Eq. (4) in the paper
		label_loss_one = self.xent(label_scores_one, labels.squeeze())
		label_loss_two = self.xent(label_scores_two, labels.squeeze())
		# GNN loss, Eq. (10) in the paper
		gnn_loss2 = self.xent(gnn_scores2, labels.squeeze())
		# the loss function of Rio-GNN, Eq. (11) in the paper
		final_loss = gnn_loss2 + self.lambda_1 * label_loss_one
		#final_loss = gnn_loss2 + (label_loss_one + label_loss_two)

		return final_loss