from utils.utils import sparse_to_adjlist
from scipy.io import loadmat

"""
	Read data and save the adjacency matrices to adjacency lists
	Paper: Reinforced Neighborhood Selection Guided Multi-Relational Graph Neural Networks
	Source: https://github.com/safe-graph/RioGNN
"""


if __name__ == "__main__":

	prefix = './data/'

	# Yelp
	yelp = loadmat('data/YelpChi.mat')
	net_rur = yelp['net_rur']
	net_rtr = yelp['net_rtr']
	net_rsr = yelp['net_rsr']
	yelp_homo = yelp['homo']

	sparse_to_adjlist(net_rur, prefix + 'yelp_rur_adjlists.pickle')
	sparse_to_adjlist(net_rtr, prefix + 'yelp_rtr_adjlists.pickle')
	sparse_to_adjlist(net_rsr, prefix + 'yelp_rsr_adjlists.pickle')
	sparse_to_adjlist(yelp_homo, prefix + 'yelp_homo_adjlists.pickle')

	# Amazon
	amz = loadmat('data/Amazon.mat')
	net_upu = amz['net_upu']
	net_usu = amz['net_usu']
	net_uvu = amz['net_uvu']
	amz_homo = amz['homo']

	sparse_to_adjlist(net_upu, prefix + 'amz_upu_adjlists.pickle')
	sparse_to_adjlist(net_usu, prefix + 'amz_usu_adjlists.pickle')
	sparse_to_adjlist(net_uvu, prefix + 'amz_uvu_adjlists.pickle')
	sparse_to_adjlist(amz_homo, prefix + 'amz_homo_adjlists.pickle')

	# Mimic
	mic = loadmat('data/Mimic.mat')
	rel_vav = mic['rel_vav']
	rel_vdv = mic['rel_vdv']
	rel_vmv = mic['rel_vmv']
	rel_vpv = mic['rel_vpv']
	mic_homo = mic['homo']

	sparse_to_adjlist(rel_vav, prefix + 'mic_vav_adjlists.pickle')
	sparse_to_adjlist(rel_vdv, prefix + 'mic_vdv_adjlists.pickle')
	sparse_to_adjlist(rel_vmv, prefix + 'mic_vmv_adjlists.pickle')
	sparse_to_adjlist(rel_vpv, prefix + 'mic_vpv_adjlists.pickle')
	sparse_to_adjlist(mic_homo, prefix + 'mic_homo_adjlists.pickle')
