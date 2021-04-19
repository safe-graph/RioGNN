import os
import argparse
import time
from sklearn.model_selection import train_test_split

from utils import *
from model import *
from layers import *
from graphsage import *
from actor_critic import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
	Training Rio-GNN
	Paper: Reinforced Neighborhood Selection Guided Multi-Relational Graph Neural Networks
	Source: https://github.com/safe-graph/RioGNN
"""

parser = argparse.ArgumentParser()

# dataset and model dependent args
parser.add_argument('--data', type=str, default='yelp', help='The dataset name. [yelp, amazon]')
parser.add_argument('--model', type=str, default='Rio', help='The model name. [Rio, SAGE]')
parser.add_argument('--inter', type=str, default='GNN', help='The inter-relation aggregator type. [Att, Weight, Mean, GNN]')
parser.add_argument('--batch-size', type=int, default=1024, help='Batch size 1024 for yelp, 256 for amazon.')

# hyper-parameters
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--lambda_1', type=float, default=2, help='Simi loss weight.')
parser.add_argument('--lambda_2', type=float, default=1e-3, help='Weight decay (L2 loss weight).')
parser.add_argument('--emb-size', type=int, default=64, help='Node embedding size at the last layer.')
parser.add_argument('--num-epochs', type=int, default=400, help='Number of epochs.')
parser.add_argument('--test-epochs', type=int, default=3, help='Epoch interval to run test set.')
parser.add_argument('--under-sample', type=int, default=1, help='Under-sampling scale.')
parser.add_argument('--step-size', type=float, default=1e-1, help='RL action step size')

# other args
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')

# RL args
parser.add_argument('--device', type=str, default="cpu", help='"cuda" if torch.cuda.is_available() else "cpu".')
parser.add_argument('--GAMMA', type=float, default=0.95, help='Actor discount factor.')
parser.add_argument('--LR', type=float, default=0.01, help='Actor learning rate.')
parser.add_argument('--stop_num', type=int, default=3, help='Deep switching or termination conditions.')
parser.add_argument('--ALPHA', type=list, default=[10, 10, 10], help='Adjustment parameters for depth and width.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("not args.no_cuda:  "+str(not args.no_cuda))
print("torch.cuda.is_available():  "+str(torch.cuda.is_available()))
print(f'run on {args.data}')

# load graph, feature, and label
[homo, relation1, relation2, relation3], feat_data, labels = load_data(args.data)

# train_test split
np.random.seed(args.seed)
random.seed(args.seed)
if args.data == 'yelp':
	index = list(range(len(labels)))
	idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=0.60,
															random_state=2, shuffle=True)
elif args.data == 'amazon':  # amazon
	# 0-3304 are unlabeled nodes
	index = list(range(3305, len(labels)))
	idx_train, idx_test, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
															test_size=0.60, random_state=2, shuffle=True)

# split pos neg sets for under-sampling
train_pos, train_neg = pos_neg_split(idx_train, y_train)

# initialize model input
features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
feat_data = normalize(feat_data)
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
if args.cuda:
	features.cuda()

# set input graph
if args.model == 'SAGE':
	adj_lists = homo
else:
	adj_lists = [relation1, relation2, relation3]

print(f'Model: {args.model}, Inter-AGG: {args.inter}, emb_size: {args.emb_size}.')

# initialize RL
print(len(relation1))
print(len(relation2))
print(len(relation3))
max_nodes_num = [0, 0, 0]
for i in relation1:
	if len(relation1[i]) > max_nodes_num[0]:
		max_nodes_num[0] = len(relation1[i])
for i in relation2:
	if len(relation2[i]) > max_nodes_num[1]:
		max_nodes_num[1] = len(relation2[i])
for i in relation3:
	if len(relation3[i]) > max_nodes_num[2]:
		max_nodes_num[2] = len(relation3[i])
max_nodes_wei = []
print(max_nodes_num)
for i, s in enumerate(max_nodes_num):
	length = 0
	number = s
	while number != 0:
		length += 1
		number = number // args.ALPHA[i]
	max_nodes_wei.append(length)

print(max_nodes_wei)
width_rl = args.ALPHA
height_rl = max_nodes_wei
actor = [[Actor(1, width_rl[i], args.device, args.LR) for j in range(height_rl[i])] for i in range(3)]
critic = [[Critic(1, width_rl[i], args.device, args.LR, args.GAMMA) for j in range(height_rl[i])] for i in range(3)]
init_rl = [0, 0, 0]
init_action = [0, 0, 0]
auc = 0

# build one-layer models
if args.model == 'Rio':
	intra1 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	intra2 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	intra3 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	inter1 = InterAgg(actor, critic, init_rl, init_action, auc, args.stop_num, width_rl, features, feat_data.shape[1], args.emb_size, adj_lists, [intra1, intra2, intra3], inter=args.inter,
					  step_size=args.step_size, cuda=args.cuda)
elif args.model == 'SAGE':
	agg1 = MeanAggregator(features, cuda=args.cuda)
	enc1 = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg1, gcn=True, cuda=args.cuda)

if args.model == 'Rio':
	gnn_model = OneLayerRio(2, inter1, args.lambda_1)
elif args.model == 'SAGE':
	# the vanilla GraphSAGE model as baseline
	enc1.num_samples = 5
	gnn_model = GraphSage(2, enc1)

if args.cuda:
	gnn_model.cuda()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr, weight_decay=args.lambda_2)
times = []
performance_log = []

#
#gnn_auc = 0
gnn_auc_train = 0
stop_flag = True
start_all_time = time.time()
end_epoch = 0
# train the model
for epoch in range(args.num_epochs):
	# randomly under-sampling negative nodes for each epoch
	sampled_idx_train = undersample(train_pos, train_neg, scale=1)
	rd.shuffle(sampled_idx_train)

	# send number of batches to model to let the RLModule know the training progress
	num_batches = int(len(sampled_idx_train) / args.batch_size) + 1

	if args.model == 'Rio':
		inter1.batch_num = num_batches
		inter1.auc = gnn_auc_train

	loss = 0.0
	epoch_time = 0

	# mini-batch training
	for batch in range(num_batches):
		start_time = time.time()
		i_start = batch * args.batch_size
		i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
		batch_nodes = sampled_idx_train[i_start:i_end]
		batch_label = labels[np.array(batch_nodes)]
		optimizer.zero_grad()
		if args.cuda:
			loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
		else:
			loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))
		loss.backward()
		optimizer.step()
		end_time = time.time()
		epoch_time += end_time - start_time
		loss += loss.item()

	print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')

	# testing the model for every $test_epoch$ epoch
	if epoch % args.test_epochs == 0:
		if args.model == 'SAGE':
			test_sage(idx_test, y_test, gnn_model, args.batch_size)
		else:
			gnn_auc, label_auc, gnn_recall, label_recall = test_rio(idx_test, y_test, gnn_model, args.batch_size)
			performance_log.append([gnn_auc, label_auc, gnn_recall, label_recall])
	gnn_auc_train = test_rio_train(idx_train, y_train, gnn_model, args.batch_size)
	stop_flag = inter1.RL
	if stop_flag == True:
		end_epoch = epoch


end_all_time = time.time()
total_epoch_time = end_all_time - start_all_time
print(f'Total time spent: {total_epoch_time}s')
print(f'Total epoch: {end_epoch}')