import math
import numpy as np

# SPIREL
class Params:
	def __init__(self):
		# dataset
		self.data_path = "gowalla"
		#self.data_path = "taxi"
		#self.data_path = "yelp"
		#self.data_path = "foursquare"

		# hyper parameters
		self.emb_dim = 40  # size of latent dimensions
		self.epoches = 10  # number of max iteration

		self.learning_rate = 0.01
		self.user_reg = 0.0001
		self.item_reg = 0.0001
		self.epsilon = 0.8  # total privacy budget
		self.budget_ratio = 0.5  # budget_ration for transition matrix, (1 - budget_ratio) for perturbing gradients
		
		# for Adam optimizer
		self.beta_1 = 0.9
		self.beta_2 = 0.998
		self.adam_eps = 0.000000001

		# transition matrix construction (max gpu allocation)
		self.transition_batch_size = 500
		
		# evaluation
		self.top_k = 5	