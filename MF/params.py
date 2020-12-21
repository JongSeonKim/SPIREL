# matrix Factorization & Privacy Enhanced Matrix Factorization
import math

class Params:
	def __init__(self):
		self.data_path = "gowalla"
		#self.data_path = "taxi"
		#self.data_path = "yelp"
		#self.data_path = "foursquare"
		
		# common parameters
		self.emb_dim = 40 
		self.epoches = 20 
		self.epsilon = 1.0

		# learning parameters
		self.reg = 0.0001  # regularization parameters
		
		self.learning_rate = 0.1

		# for evaluation
		self.top_k = 5

		# for dimension projection
		self.proj_dim = 50