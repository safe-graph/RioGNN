import os
import argparse
from time import localtime, strftime, time
from sklearn.model_selection import train_test_split

from utils.utils import *
from model.model import *
from model.layers import *
from model.graphsage import *
from RL.rl_model import *
"""
    Training and testing RIO-GNN
    Paper: Reinforced Neighborhood Selection Guided Multi-Relational Graph Neural Networks
    Source: https://github.com/safe-graph/RioGNN
"""

parser = argparse.ArgumentParser()

# dataset and model dependent args
parser.add_argument('--data', type=str, default='amazon', help='The dataset name. [yelp, amazon, mimic]')
parser.add_argument('--log_path', default='log/', type=str, help="Path of results")
parser.add_argument('--model', type=str, default='RIO', help='The model name. [RIO, SAGE]')
parser.add_argument('--inter', type=str, default='GNN',
                    help='The inter-relation aggregator type. [Att, Weight, Mean, GNN]')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Batch size 1024 for yelp, 256 for amazon, X for mimic.')

# hyper-parameters
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--lambda_1', type=float, default=2, help='Simi loss weight.')
parser.add_argument('--lambda_2', type=float, default=1e-3, help='Weight decay (L2 loss weight).')
parser.add_argument('--emb_size', type=int, default=64, help='Node embedding size at the last layer.')
parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs.')
parser.add_argument('--test_epochs', type=int, default=3, help='Epoch interval to run test set.')
parser.add_argument('--test_ratio', type=float, default=0.60, help='Test set size.')
parser.add_argument('--under_sample', type=int, default=1, help='Under-sampling scale.')

# other args
parser.add_argument('--use_cuda', default=False, action='store_true', help='Training with CUDA.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')

# RL args
parser.add_argument('--device', type=str, default="cpu", help='"cuda" if torch.cuda.is_available() else "cpu".')
parser.add_argument('--GAMMA', type=float, default=0.95, help='Actor discount factor.')
parser.add_argument('--LR', type=float, default=0.01, help='Actor learning rate.')
parser.add_argument('--stop_num', type=int, default=3, help='Deep switching or termination conditions.')
parser.add_argument('--ALPHA', type=int, default=10, help='Adjustment parameters for depth and width.')

if __name__ == '__main__':

    print('\n+------------------------------------------------------------------------------------------+\n'
          '* Training and testing RIO-GNN *\n'
          '* Paper: Reinforced Neighborhood Selection Guided Multi-Relational Graph Neural Networks *\n'
          '* Source: https://github.com/safe-graph/RioGNN *\n'
          '\n+------------------------------------------------------------------------------------------+\n', flush=True
          )

    # load hyper-parameters
    args = parser.parse_args()

    # generate log folder
    log_save_path = args.log_path + 'log_' + strftime("%m%d%H%M%S", localtime())
    os.mkdir(log_save_path)
    print("Log save path:  ", log_save_path, flush=True)

    # device
    args.cuda = args.use_cuda and torch.cuda.is_available()
    print("CUDA:  " + str(args.cuda), flush=True)

    # load graph, feature, and label
    homo, relations, feat_data, labels, index = load_data(args.data)
    print("Running on:  " + str(args.data), flush=True)
    print("The number of relations:  " + str(len(relations)), flush=True)

    # train_test split
    np.random.seed(args.seed)
    random.seed(args.seed)
    idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels,
                                                            test_size=args.test_ratio, random_state=2, shuffle=True)

    # split pos neg sets for under-sampling
    train_pos, train_neg = pos_neg_split(idx_train, y_train)

    # initialize model input
    features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
    feat_data = normalize(feat_data)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    if args.cuda:
        features.cuda()

    # initialize RL action space
    width_rl = [args.ALPHA for r in range(len(relations))]
    height_rl = [math.ceil(pow(len(max(relations[r].values(), key=len)), 1 / width_rl[r]))
                 for r in range(len(relations))]
    print('Width of each relation tree:  ' + str(width_rl), flush=True)
    print('Height of each relation tree:  ' + str(height_rl), flush=True)

    # build one-layer models
    print('Model:  {0}, Inter-AGG:  {1}, emb_size:  {2}.'.format(args.model, args.inter, args.emb_size))
    if args.model == 'RIO':
        adj_lists = relations
        intra_aggs = [IntraAgg(features, feat_data.shape[1], cuda=args.cuda) for r in range(len(relations))]
        inter1 = InterAgg(width_rl, height_rl, args.device, args.LR, args.GAMMA, args.stop_num,
                          features, feat_data.shape[1],
                          args.emb_size, adj_lists,
                          intra_aggs, inter=args.inter,
                          cuda=args.cuda)
        gnn_model = OneLayerRio(2, inter1, args.lambda_1)
    elif args.model == 'SAGE':
        adj_lists = homo
        agg1 = MeanAggregator(features, cuda=args.cuda)
        enc1 = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg1, gcn=True, cuda=args.cuda)
        # the vanilla GraphSAGE model as baseline
        enc1.num_samples = 5
        gnn_model = GraphSage(2, enc1)

    if args.cuda:
        gnn_model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr,
                                 weight_decay=args.lambda_2)

    gnn_auc_train = 0
    start_all_time = time()

    # train the model
    for epoch in range(args.num_epochs):
        print('\n+------------------------------------------------------------------------------------------+\n'
              '                                        Epoch {0}                                               '
              '\n+------------------------------------------------------------------------------------------+\n'.
              format(epoch), flush=True
              )
        # randomly under-sampling negative nodes for each epoch
        sampled_idx_train = undersample(train_pos, train_neg, scale=args.under_sample)
        rd.shuffle(sampled_idx_train)

        # send number of batches to model to let the RLModule know the training progress
        num_batches = int(len(sampled_idx_train) / args.batch_size) + 1
        if args.model == 'RIO':
            inter1.batch_num = num_batches
            inter1.auc = gnn_auc_train

        loss = 0.0
        epoch_time = 0

        # mini-batch training
        for batch in range(num_batches):
            start_time = time()
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
            end_time = time()
            epoch_time += end_time - start_time
            loss += loss.item()

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', flush=True)
        print('Loss: {0}, time: {1}s'.format(loss.item() / num_batches, epoch_time), flush=True)

        # testing the model for every $test_epoch$ epoch
        if epoch % args.test_epochs == 0:
            if args.model == 'SAGE':
                test_sage(idx_test, y_test, gnn_model, args.batch_size)
            else:
                gnn_auc, label_auc, gnn_recall, label_recall = test_rio(idx_test, y_test, gnn_model, args.batch_size)
        gnn_auc_train = test_rio_train(idx_train, y_train, gnn_model, args.batch_size)

        # termination
        if not inter1.RL:
            break

    # log
    with open(log_save_path + '/thresholds_log.txt', 'w') as file:
        for l in inter1.rl_tree.thresholds_log:
            file.writelines(str(l) + '\n')
    with open(log_save_path + '/states_log.txt', 'w') as file:
        for l in inter1.rl_tree.states_log:
            file.writelines(str(l) + '\n')

    # end
    print('\n+------------------------------------------------------------------------------------------+\n')
    end_all_time = time()
    total_epoch_time = end_all_time - start_all_time
    print('Total time spent:  ' + str(total_epoch_time), flush=True)
    print('Total epoch:  ' + str(epoch), flush=True)
