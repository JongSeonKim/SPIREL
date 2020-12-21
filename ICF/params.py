# ICF
class Params:
	def __init__(self):
		# Dataset path
		self.data_path = "gowalla"
		#self.data_path = "taxi"
		#self.data_path = "yelp"
		#self.data_path = "foursquare"
		
		# Privacy budget
		self.epsilon = 0.8
		
		# similiarity scaling factor
		self.alpha = 0.5
		
		# Full evaluation batch size
		self.joint_batch = 2
		self.batch_size = 40000 # for evaluation
		
		# Hit Ratio top-k
		self.top_k = 5