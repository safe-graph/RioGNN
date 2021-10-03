from operator import itemgetter
from RL.actor_critic import *

"""
    RL Forest.
    Paper: Reinforced Neighborhood Selection Guided Multi-Relational Graph Neural Networks
    Source: https://github.com/safe-graph/RioGNN
"""


class RLForest:

    def __init__(self, width_rl, height_rl, device, LR, GAMMA, stop_num, r_num):
        """
        Initialize the RL Forest.
        :param width_rl: width of each relation tree
        :param height_rl: height of each relation tree
        :param device: "cuda" / "cpu"
        :param LR: Actor learning rate (hyper-parameters of AC)
        :param GAMMA: Actor discount factor (hyper-parameters of AC)
        :param stop_num: deep switching or termination conditions
        :param r_num: the number of relations
        """

        self.actors = [[Actor(1, width_rl[r], device, LR) for j in range(height_rl[r])]
                       for r in range(r_num)]
        self.critics = [[Critic(1, width_rl[r], device, LR, GAMMA) for j in range(height_rl[r])]
                        for r in range(r_num)]
        self.r_num = r_num

        # current RLT depth for each relation
        self.init_rl = [0 for r in range(r_num)]
        # number of epochs performed at the current depth for each relation
        self.init_termination = [0 for r in range(r_num)]
        # action interval of current depth for each relation
        self.init_action = [0 for r in range(r_num)]

        # backtracking
        self.max_auc = 0
        self.max_thresholds = [0 for r in range(r_num)]

        # termination and boundary conditions
        self.width = list(width_rl)
        self.stop_num = stop_num

        # log
        self.thresholds_log = []
        self.actions_log = []
        self.states_log = []
        self.scores_log = []
        self.rewards_log = []

    def get_threshold(self, scores, labels, previous_thresholds, batch_num, auc):
        """
        The reinforcement learning module.
        It updates the neighbor filtering threshold for each relation based
        on the average neighbor distances between two consecutive epochs.
        :param scores: the neighbor nodes label-aware scores for each relation
        :param labels: the batch node labels used to select positive nodes
        :param previous_thresholds: the current neighbor filtering thresholds for each relation
        :param batch_num: numbers batches in an epoch
        :param auc: the auc of the previous filter thresholds for each relation
        """

        new_scores = get_scores(scores, labels)
        rl_flag0 = 0

        # during the epoch
        if len(self.scores_log) % batch_num != 0 or len(self.scores_log) < batch_num:

            # do not call RL module within the epoch or within the first two epochs
            new_thresholds = list(previous_thresholds)

        # after completing each epoch
        else:

            # STATE
            # get current states according to average scores
            # Eq.(8) in the paper
            current_epoch_states = [sum(s) / batch_num for s in zip(*self.scores_log[-batch_num:])]
            new_states = [np.array([s], float) for i, s in enumerate(current_epoch_states)]

            # backtracking
            if auc >= self.max_auc:
                self.max_auc = auc
                self.max_thresholds = list(previous_thresholds)

            new_actions = [0 for r in range(self.r_num)]
            new_thresholds = [0 for r in range(self.r_num)]

            # the first epoch
            if len(self.states_log) == 0:
                # update the record of the number of epochs in the current depth
                self.init_termination = [i + 1 for i in self.init_termination]
                # ACTION
                # get current actions for current states
                # Eq.(11) in the paper
                for r_num in range(self.r_num):
                    new_actions[r_num], new_thresholds[r_num] = self.get_action(new_states, r_num)

            # after the first epoch
            else:
                # STATE
                # get previous states
                previous_states = self.states_log[-1]
                # ACTION
                # get previous actions
                previous_actions = self.actions_log[-1]

                # REWARD
                # compute reward for each relation
                # Eq. (9) in the paper
                new_rewards = [s if 0 < previous_thresholds[i] and previous_thresholds[i] <= 1 else -100 for i, s in
                               enumerate(current_epoch_states)]

                # determine whether to enter the next depth
                r_flag = self.adjust_depth()

                # after the smallest continuous epoch
                for r_num in range(self.r_num):

                    # go to the next depth
                    if r_flag[r_num] == 1:

                        if len(self.actors[r_num]) == self.init_rl[r_num] + 1:
                            # relation tree remains unchanged after converging
                            self.init_termination[r_num] = self.init_termination[r_num]
                            # ACTION
                            new_actions[r_num] = previous_actions[r_num]
                            new_thresholds[r_num] = self.max_thresholds[r_num]
                            rl_flag0 += 1
                            print("Relation {0} is complete ！！！！!".format(str(r_num + 1)), flush=True)

                        else:
                            # update the parameter space when entering the next depth
                            # Eq. (7) in the paper
                            self.init_termination[r_num] = 0
                            self.init_rl[r_num] = self.init_rl[r_num] + 1
                            self.init_action[r_num] = self.max_thresholds[r_num] - (self.width[r_num] / 2) * \
                                                      pow(1 / self.width[r_num], self.init_rl[r_num] + 1)
                            # ACTION
                            # Eq. (11) in the paper
                            new_actions[r_num], new_thresholds[r_num] = self.get_action(new_states, r_num)

                    # keep current depth
                    else:
                        self.init_termination[r_num] = self.init_termination[r_num] + 1
                        # POLICY
                        # Eq. (10) in the paper
                        self.learn(previous_states, previous_actions, new_states, new_rewards, r_num)
                        # ACTION
                        # Eq. (11) in the paper
                        new_actions[r_num], new_thresholds[r_num] = self.get_action(new_states, r_num)

                self.rewards_log.append(new_rewards)
                print('Rewards:  ' + str(new_rewards), flush=True)

            self.states_log.append(new_states)
            print('States:  ' + str(new_states), flush=True)
            self.thresholds_log.append(new_thresholds)
            print('Thresholds:  ' + str(new_thresholds), flush=True)
            self.actions_log.append(new_actions)

        self.scores_log.append(new_scores)

        print("Historical maximum AUC:  " + str(self.max_auc), flush=True)
        print("Thresholds to obtain the historical maximum AUC:  " + str(self.max_thresholds), flush=True)
        print('Current depth of each RL Tree:  ' + str(self.init_rl), flush=True)

        # RLF termination
        rl_flag = False if rl_flag0 == self.r_num else True
        print('Completion flag of the entire RL Forest:  ' + str(rl_flag), flush=True)

        return new_thresholds, rl_flag

    def learn(self, previous_states, previous_actions, new_states, new_rewards, r_num):
        """
        :param previous_states: the previous states
        :param previous_actions: the previous actions
        :param new_states: the current states
        :param new_rewards: the current rewards
        :param r_num: the index of relation
        """

        td_error = self.critics[r_num][self.init_rl[r_num]].train_Q_network(previous_states[r_num],
                                                                            new_rewards[r_num],
                                                                            new_states[r_num])
        self.actors[r_num][self.init_rl[r_num]].learn(previous_states[r_num],
                                                      previous_actions[r_num],
                                                      td_error)
        return

    def get_action(self, new_states, r_num):
        """
        :param new_states: the current states
        :param r_num: the index of relation
        :returns: new actions and thresholds for new_states under relation r_num
        """

        new_actions = self.actors[r_num][self.init_rl[r_num]].choose_action(new_states[r_num])
        new_thresholds = self.init_action[r_num] + (new_actions + 1) * \
                         pow(1 / self.width[r_num], self.init_rl[r_num] + 1)
        new_thresholds = 1 if new_thresholds >= 1 else new_thresholds

        return new_actions, new_thresholds

    def adjust_depth(self):
        """
        :returns: the depth flag of each relation
        """

        r_flag = [1 for r in range(self.r_num)]
        for r_num in range(self.r_num):
            if self.init_termination[r_num] > self.stop_num:
                for s in range(self.stop_num - 1):
                    r_flag[r_num] = r_flag[r_num] * (
                        1 if self.actions_log[-1 * (s + 1)][r_num] == self.actions_log[-1 * (s + 2)][r_num] else 0
                    )
            else:
                r_flag[r_num] = 0

        return r_flag


def get_scores(scores, labels):
    """
    Get the scores of current batch.
    :param scores: the neighbor nodes label-aware scores for each relation
    :param labels: the batch node labels used to select positive nodes
    :returns: the state of current batch
    """

    relation_scores = []

    # only compute the average neighbor distances for positive nodes
    pos_index = (labels == 1).nonzero().tolist()
    pos_index = [i[0] for i in pos_index]

    # compute average neighbor distances for each relation
    for score in scores:
        pos_scores = itemgetter(*pos_index)(score)
        neigh_count = sum([1 if isinstance(i, float) else len(i) for i in pos_scores])
        pos_sum = [i if isinstance(i, float) else sum(i) for i in pos_scores]
        relation_scores.append(sum(pos_sum) / neigh_count)

    return relation_scores
