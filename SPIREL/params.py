import math
import numpy as np

# SPIREL
class Params:
	def __init__(self):
		# dataset
		#self.data_path = "taxi_checkin"
		#self.data_path = "taxi_medium"
		self.data_path = "gowalla_LA"
		#self.data_path = "yelp_RAPPOR"
		#self.data_path = "foursquare-1M_RAPPOR"

		# hyper parameters
		# SPIREL taxi_checkin = 40, taxi_medium = 70, gowalla_LA = 40, yelp_RAPPOR = 40, foursquare-1M = 40
		# SPIREL_NP taxi_checkin = 40, taxi_medium = 70, gowalla_LA = 40, yelp_RAPPOR = 40, foursquare-1M = 40
		self.emb_dim = 40  # size of latent dimensions
		self.epoches = 20  # number of max iteration
		# SPIREL taxi_checkin = 0.01, taxi_medium = 0.01, gowalla_LA = 0.01, yelp_RAPPOR = 0.01, foursquare-1M = 0.001
		# SPIREL_NP taxi_checkin = 0.01, taxi_medium = 0.01, gowalla_LA = 0.01, yelp_RAPPOR = 0.01, foursquare-1M = 0.001
		# SPIREL PM taxi_checkin = 0.001
		self.learning_rate = 0.001
		# SPIREL taxi_checkin = 0.0001, taxi_medium = 0.000001, gowalla_LA = 0.0000001, yelp_RAPPOR = 0.001, foursquare-1M = 0.0001
		# SPIREL_NP taxi_checkin = 0.0001, taxi_medium = 0.001, gowalla_LA = 0.001, yelp_RAPPOR = 0.001, foursquare-1M = 0.001
		self.user_reg = 0.0000001
		# SPIREL taxi_checkin = 0.0001, taxi_medium = 0.000001, gowalla_LA = 0.0000001, yelp_RAPPOR = 0.001, foursquare-1M = 0.0001
		# SPIREL_NP taxi_checkin = 0.0001, taxi_medium = 0.001, gowalla_LA = 0.001, yelp_RAPPOR = 0.001, foursquare-1M = 0.001
		self.item_reg = 0.0000001
		self.epsilon = 1.0  # total privacy budget
		self.budget_ratio = 0.5  # budget_ration for transition matrix, (1 - budget_ratio) for perturbing gradients
		
		# for Adam optimizer
		self.beta_1 = 0.9
		self.beta_2 = 0.998
		self.adam_eps = 0.000000001

		# transition matrix construction
		# gowalla_LA : 2500, taxi_checkin = 2500, taxi_medium = 1000, yelp_RAPPOR = 500, foursquare-1M_RAPPOR = 200
		self.transition_batch_size = 2500
		
		# evaluation
		self.top_k = 10
		
		# output path
		#self.output_path = self.data_path + "-" + str(self.epsilon) + "-" + str(self.epoches) + "-" + str(self.emb_dim) + "-" + str(self.top_k)
		self.output_path = self.data_path + "-" + str(self.epsilon) + "-" + str(self.emb_dim) + "-" + str(self.top_k)
		self.np_output_path = self.data_path + "-" + str(self.emb_dim) + "-" + str(self.top_k)