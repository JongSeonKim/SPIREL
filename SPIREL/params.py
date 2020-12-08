import math
import numpy as np

# SPIREL
class Params:
	#_dataset = "gowalla_LA"
	#_dataset = "taxi_medium"
	#_dataset = "taxi_checkin"
	self.dataset = "yelp_RAPPOR"

	# common parameters
	_k = 10  # size of latent dimensions
	_max_iter = 10  # number of max iteration
	_gamma = 1  # learning rate
	_lambda = 0.001  # regularization parameter
	_eps = 0.8  # total privacy budget
	_eps_split = 0.5  # budget for transition matrix, (1 - splitfrac) for perturbing gradients

	# for trie construction
	_user_per_gpu = 2000 # gowalla_LA = 5000, taxi = 7500, yelp = 2000
	_orr_prob = np.float32(1 / (math.exp((_eps * _eps_split)) + 1))
	_eta = 4  # Threshold

	# for Adam optimizer
	_beta_1 = 0.9
	_beta_2 = 0.999
	_adam_eps = 0.000000001

	# for Recommendation
	_top_k = 7