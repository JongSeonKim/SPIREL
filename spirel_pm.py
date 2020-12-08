import numpy as np
import random
import math
import tensorflow as tf
import time

from numpy import array
from params import Params

class SPIREL_PM(object):
	def __init__(self, dataset, params, sess):
		# dataset & session
		[self.user_train, self.user_test, self.user_num, self.item_num] = dataset
		self.output_path = params.output_path
		self._sess = sess
		
		# hyper-parameter
		self.emb_dim = params.emb_dim  # size of latent dimensions
		self.epoches = params.epoches  # number of max iteration
		self.learning_rate = params.learning_rate
		self.user_reg = params.user_reg
		self.item_reg = params.item_reg
		self.transition_batch_size = params.transition_batch_size
		
		# privacy parameter
		self.epsilon = params.epsilon # total privacy budget
		self.budget_ratio = params.budget_ratio # budget for perturbing transition matrix & gradients
		self.orr_prob = np.float32(1 / (math.exp((self.epsilon * self.budget_ratio)) + 1))
		self.pm_c = np.float32((math.exp(self.epsilon * self.budget_ratio / 2) + 1) / (math.exp(self.epsilon * self.budget_ratio / 2) - 1))
		self.pm_prob = np.float32(math.exp(self.epsilon * self.budget_ratio / 2) / (math.exp(self.epsilon * self.budget_ratio / 2) + 1))
		
		# Adam optimizer
		self.beta_1 = params.beta_1
		self.beta_2 = params.beta_2
		self.adam_eps = params.adam_eps
		
		# evaluation paramater
		self.top_k = params.top_k
		self.best_hr = tf.compat.v1.Variable(-1.0, dtype=tf.float32, name='best_hr')
		self.best_mrr = tf.compat.v1.Variable(-1.0, dtype=tf.float32, name='best_mrr')
		self.best_iter = tf.compat.v1.Variable(0, dtype=tf.int32, name='best_iter')
		
		# input
		self._user_item_matrix = None
		self._item_item_matrix = None
		
		# training nodes
		with tf.name_scope("training"):
			self._batch_loss = None
			self._reg_loss = None
			self._user_item_loss = None
			self._item_item_loss = None
			self._train_op = None
			self._global_step = tf.compat.v1.Variable(0, name="global_step", trainable=False)
			
		# build graph
		self.build_transition_matrix()
		self.build_graph()
		self.build_eval_graph()
		
	def build_transition_matrix(self):
		with tf.name_scope("transition_input"):
			pattern_indices = tf.compat.v1.placeholder(tf.int32, name='pattern_indices')
			self._pattern_indices = pattern_indices
			
		self.perturb_transition_matrix(pattern_indices)
	
	def perturb_transition_matrix(self, pattern_indices):
		# calculate
		pair_num = tf.shape(pattern_indices)[0]
		# build transition matrix
		prob_matrix = tf.random.uniform([pair_num, self.item_num, self.item_num], minval=0.0, maxval=1.0)
	
		valid_prob = tf.gather_nd(prob_matrix, pattern_indices)
		valid_res = tf.cast(valid_prob < self.orr_prob, tf.float32)
		
		values = tf.ones([pair_num])
		prob_matrix = tf.tensor_scatter_nd_add(prob_matrix, pattern_indices, values)
		invalid_res = tf.cast(prob_matrix < 0.5, tf.float32)
		invalid_res = tf.tensor_scatter_nd_add(invalid_res, pattern_indices, valid_res)
		
		res = tf.reduce_sum(invalid_res, axis=0)
		self._res = res
	
	def build_graph(self):
		with tf.name_scope("matrix_input"):
			user_item_matrix = tf.compat.v1.placeholder(tf.float32, name='user_item_matrix')
			item_item_matrix = tf.compat.v1.placeholder(tf.float32, name='item_item_matrix')
			self._user_item_matrix, self._item_item_matrix = user_item_matrix, item_item_matrix
			
		with tf.name_scope("batch_input"):	
			cur_user = tf.compat.v1.placeholder(tf.int32, name='cur_user')
			cur_item = tf.compat.v1.placeholder(tf.int32, name='cur_item')
			user_item_indices = tf.compat.v1.placeholder(tf.int32, name='user_item_indices')
			item_item_indices = tf.compat.v1.placeholder(tf.int32, name='item_item_indices')
			item_ind = tf.compat.v1.placeholder(tf.int32, name='item_ind')
			emb_dim_ind = tf.compat.v1.placeholder(tf.int32, name='emb_dim_ind')
			grad_ind = tf.compat.v1.placeholder(tf.int32, name='grad_ind')
			self._cur_user, self._cur_item, self._user_item_indices, self._item_item_indices, self._item_ind, self._emb_dim_ind, self._grad_ind = \
				cur_user, cur_item, user_item_indices, item_item_indices, item_ind, emb_dim_ind, grad_ind
				
		with tf.name_scope("optimize_input"):
			user_als_term = tf.compat.v1.placeholder(tf.float32, name='user_als_term')
			perturbed_grads = tf.compat.v1.placeholder(tf.float32, name='perturbed_grads')
			pm_item_grads = tf.compat.v1.placeholder(tf.float32, name='pm_item_grads')
			pm_item_loss = tf.compat.v1.placeholder(tf.float32, name='pm_item_loss')
			pm_reg_loss = tf.compat.v1.placeholder(tf.float32, name='pm_reg_loss')
			self._user_als_term, self._perturbed_grads, self._pm_item_grads = user_als_term, perturbed_grads, pm_item_grads
			self._pm_item_loss, self._pm_reg_loss = pm_item_loss, pm_reg_loss
		
		# calculate
		user_item_pair = tf.shape(user_item_indices)[0]
		item_item_pair = tf.shape(item_item_indices)[0]
		
		user_item_pred, item_item_pred = self.forward(cur_user, cur_item)
		
		user_item_loss, item_item_loss, user_item_loss_l2, item_item_loss_l2, batch_loss, reg_loss = \
			self.loss(cur_user, cur_item, user_item_pred, item_item_pred, user_item_pair, item_item_pair)
		self._user_item_loss, self._item_item_loss, self._user_item_loss_l2, self._item_item_loss_l2, self._batch_loss, self._reg_loss = \
			user_item_loss, item_item_loss, user_item_loss_l2, item_item_loss_l2, batch_loss, reg_loss
		
		user_term, user_grads, item_grads = self.get_grads(cur_user, user_item_loss, item_item_loss, reg_loss)
		self._user_term, self._user_grads, self._item_grads = user_term, user_grads, item_grads
		
		self.optimize(user_als_term, perturbed_grads, pm_item_grads, item_item_loss, reg_loss)
			
	def forward(self, cur_user, cur_item):
		# model parameter
		with tf.name_scope("embedding"):
			user_embeddings = tf.compat.v1.get_variable(
				name = "user_embeddings",
				shape = [self.user_num, self.emb_dim],
				initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			self._user_embeddings = user_embeddings

			item_embeddings = tf.compat.v1.get_variable(
				name = "item_embeddings",
				shape = [self.item_num, self.emb_dim],
				initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			self._item_embeddings = item_embeddings

		return self.predict(cur_user, cur_item)
			
	def predict(self, cur_user, cur_item):
		cur_user_emb = tf.gather(self._user_embeddings, cur_user)
		cur_item_emb = tf.gather(self._item_embeddings, cur_item)
		
		user_item_pred = tf.matmul(cur_user_emb, self._item_embeddings, transpose_b=True)
		item_item_pred = tf.matmul(cur_item_emb, self._item_embeddings, transpose_b=True)
		
		return user_item_pred, item_item_pred
	
	def loss(self, cur_user, cur_item, user_item_pred, item_item_pred, user_item_pair, item_item_pair):
		cur_user_emb = tf.gather(self._user_embeddings, cur_user)
		cur_item_emb = tf.gather(self._item_embeddings, cur_item)	
		cur_user_rating = tf.gather(self._user_item_matrix, cur_user)

		user_item_loss = (cur_user_rating - user_item_pred)
		item_item_loss = (self._item_item_matrix - item_item_pred)
		
		user_item_loss_l2 = tf.nn.l2_loss(user_item_loss) / tf.cast(user_item_pair, tf.float32)
		item_item_loss_l2 = tf.nn.l2_loss(item_item_loss) / tf.cast(item_item_pair, tf.float32)
		
		batch_loss = user_item_loss_l2 + item_item_loss_l2
		reg_loss = tf.nn.l2_loss(cur_item_emb) * self.item_reg
		
		return user_item_loss, item_item_loss, user_item_loss_l2, item_item_loss_l2, batch_loss, reg_loss
	
	def get_grads(self, cur_user, user_item_loss, item_item_loss, reg_loss):
		# calculate
		cur_batch_size = tf.shape(cur_user)[0]
	
		# user embedding update
		inv_als_term = tf.linalg.inv(tf.matmul(self._item_embeddings, self._item_embeddings, transpose_a=True) + \
							tf.multiply(float(self.user_reg), tf.eye(self.emb_dim)))
		user_term = tf.matmul(tf.cast(self._user_item_matrix, tf.float32), tf.matmul(self._item_embeddings, inv_als_term))
		
		# item embedding update
		item_grads = tf.squeeze(tf.gradients(item_item_loss + reg_loss, [self._item_embeddings]))
		selected_item_loss = tf.gather_nd(user_item_loss, self._item_ind)
		user_emb = tf.gather_nd(self._user_embeddings, self._emb_dim_ind)
		user_grads = tf.multiply(selected_item_loss, user_emb)
		user_grads = tf.clip_by_value(user_grads, -1.0, 1.0)
		
		return user_term, user_grads, item_grads
	
	def optimize(self, user_als_term, perturbed_grads, pm_item_grads, item_item_loss, reg_loss):		
		cur_batch_size = tf.shape(perturbed_grads)[0]
		
		# optimize
		update_user = tf.compat.v1.assign(self._user_embeddings, user_als_term)
		self._update_user = update_user
		
		optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta_1, beta2=self.beta_2, epsilon=self.adam_eps)
		gvs = optimizer.compute_gradients(item_item_loss + reg_loss, [self._item_embeddings])
		
		total_grads = tf.tensor_scatter_nd_add(pm_item_grads, self._grad_ind, perturbed_grads) / tf.cast(cur_batch_size, tf.float32)
		
		perturbed_gradient = [(total_grads, var) for grad, var in gvs]		
		train_op = optimizer.apply_gradients(perturbed_gradient, global_step=self._global_step)
		self._train_op = train_op
		
	def train(self, is_eval):
		start = time.time()
		# input initialize
		self._sess.run(tf.compat.v1.global_variables_initializer())
		user_item_matrix, user_item_indices, item_item_indices = self.get_matrix()
		item_item_matrix = np.zeros([self.item_num, self.item_num], np.float32)
		
		transition_batch_num, remain = divmod(self.user_num, self.transition_batch_size)
		
		for b_i in range(transition_batch_num + 1):
			if b_i != transition_batch_num:
				pattern_indices = self.get_transition_matrix_batch(b_i, transition_batch_num)
			else:
				if remain != 0:
					pattern_indices = self.get_transition_matrix_batch(b_i, transition_batch_num)
			
			feed_dict = {
				self._pattern_indices: pattern_indices
			}
			
			[res] = self._sess.run(
				[self._res],
				feed_dict=feed_dict)
				
			item_item_matrix += res
	
		item_item_matrix = (item_item_matrix - self.user_num * self.orr_prob) / (0.5 - self.orr_prob)
		item_item_matrix[item_item_matrix < 0] = 1
		item_item_matrix = np.log10(item_item_matrix)
		item_item_matrix = 1 + (1.0 / (1.0 + np.exp(-item_item_matrix)))
		
		last_hr, last_mrr = 0.0, 0.0
		# learning
		update_batch_num = int(self.user_num / self.epoches)
		
		for b_i in range(self.epoches):
			users, items, item_ind, emb_dim_ind, grad_ind = self.get_batch(b_i, update_batch_num)
			
			feed_dict = {
				self._user_item_matrix: user_item_matrix,
				self._item_item_matrix: item_item_matrix,
				self._cur_user: users,
				self._cur_item: items,
				self._user_item_indices: user_item_indices,
				self._item_item_indices: item_item_indices,
				self._item_ind: item_ind,
				self._emb_dim_ind: emb_dim_ind,
				self._grad_ind: grad_ind
			}
			
			[user_term, user_grads, item_grads, user_item_loss_l2, item_item_loss_l2, item_item_loss, batch_loss, reg_loss] = self._sess.run(
				[self._user_term, self._user_grads, self._item_grads, self._user_item_loss_l2, self._item_item_loss_l2, self._item_item_loss, self._batch_loss, self._reg_loss],
				feed_dict=feed_dict)
			
			perturbed_grads = self.pm(user_grads)
			
			feed_dict = {
				self._user_als_term: user_term,
				self._perturbed_grads: perturbed_grads,
				self._pm_item_grads: item_grads,
				self._pm_item_loss: item_item_loss,
				self._pm_reg_loss: reg_loss,
				self._grad_ind: grad_ind
			}
			
			[_, _] = self._sess.run(
				[self._update_user, self._train_op],
				feed_dict=feed_dict)
			
			print("step: ", b_i + 1, " total_loss: ", batch_loss + reg_loss, " reg_loss :", reg_loss, " user_loss: " , user_item_loss_l2, " item_loss :", item_item_loss_l2)
			
			# evaluation
			if is_eval == True:
				avg_test_hr, avg_test_mrr = self.eval()
				last_hr, last_mrr = avg_test_hr, avg_test_mrr
				best_hr, best_mrr, best_iter = self.best_hr.eval(), self.best_mrr.eval(), self.best_iter.eval()

				if avg_test_hr > best_hr:
					self._sess.run([
						tf.compat.v1.assign(self.best_hr, avg_test_hr),
						tf.compat.v1.assign(self.best_mrr, avg_test_mrr),
						tf.compat.v1.assign(self.best_iter, b_i)])
						
				
		# result
		print("time :", time.time() - start)
		f = open('../result/SPIREL_PM/' + self.output_path + '.txt', 'a')
		f.write('%f	%f\n' %(round(last_hr, 4), round(last_mrr, 4)))
		f.close()
		print('best_hr:', round(self.best_hr.eval(), 4))
		print('best_mrr:', round(self.best_mrr.eval(), 4))
			
	def pm(self, user_grads):
		pm_l = (self.pm_c + 1) / 2 * user_grads - (self.pm_c - 1) / 2
		pm_r = pm_l + self.pm_c - 1
		
		prob = np.random.uniform(0.0, 1.0, np.shape(user_grads)[0])
		
		for user in range(np.shape(user_grads)[0]):
			if prob[user] < self.pm_prob:
				user_grads[user] = np.random.uniform(pm_l[user], pm_r[user])
			elif self.pm_prob <= prob[user] and prob[user] < 0.5 + self.pm_prob / 2:
				user_grads[user] = np.random.uniform(-self.pm_c, pm_l[user])
			else:
				user_grads[user] = np.random.uniform(pm_r[user], self.pm_c)
				
		return user_grads
		
	def get_matrix(self):
		user_item_indices, item_item_indices = list(), list()
		user_item_matrix = np.zeros([self.user_num, self.item_num], dtype=np.int32)

		for user in range(self.user_num):
			for i in range(len(self.user_train[user])):
				user_item_matrix[user][self.user_train[user][i][1]] += 1
				user_item_indices.append([user, self.user_train[user][i][1]])
				
				if i != len(self.user_train[user]) - 1:
					item_item_indices.append([self.user_train[user][i][1], self.user_train[user][i+1][1]])

		return user_item_matrix, user_item_indices, item_item_indices
		
	def get_transition_matrix_batch(self, b_i, batch_num):
		pattern_indices = list()
		
		if b_i != batch_num:
			users = np.arange(b_i * self.transition_batch_size, (b_i + 1) * self.transition_batch_size)
		else:
			users = np.arange(b_i * self.transition_batch_size, self.user_num)
		
		for user_ind in range(len(users)):
			position = np.random.randint(0, len(self.user_train[b_i * self.transition_batch_size + user_ind]) - 1)		
			pattern_indices.append([user_ind, 
								self.user_train[b_i * self.transition_batch_size + user_ind][position][1], 
								self.user_train[b_i * self.transition_batch_size + user_ind][position + 1][1]])
								
		return pattern_indices

	def get_batch(self, b_i, batch_num):
		item_ind, emb_dim_ind, grad_ind = list(), list(), list()
        
		if b_i != batch_num - 1:
			users = np.arange(b_i * batch_num, (b_i + 1) * batch_num)
		else:
			users = np.arange(b_i * batch_num, self.user_num)
			
		items = np.arange(self.item_num)
		
		for user_ind in range(len(users)):
			n = np.random.randint(0, self.item_num)
			d = np.random.randint(0, self.emb_dim)

			item_ind.append([user_ind, n])
			emb_dim_ind.append([user_ind, d])
			grad_ind.append([n, d])

		return users, items, item_ind, emb_dim_ind, grad_ind
		
############################################################################################################################

	def build_eval_graph(self):
		eval_user = tf.compat.v1.placeholder(dtype=tf.int32, name='eval_user')
		prev_item = tf.compat.v1.placeholder(dtype=tf.int32, name='prev_user')
		test_item = tf.compat.v1.placeholder(dtype=tf.int32, name='test_item')
		self._eval_user, self._prev_item, self._test_item = eval_user, prev_item, test_item

		eval_user_emb = tf.gather(self._user_embeddings, self._eval_user)
		prev_item_emb = tf.gather(self._item_embeddings, self._prev_item)
		test_item_emb = tf.gather(self._item_embeddings, self._test_item)
		
		user_eval_pred = tf.reduce_sum(tf.multiply(eval_user_emb, test_item_emb), axis=1)
		user_eval_pred = tf.tile(tf.expand_dims(user_eval_pred, 1), [1, self.item_num])
		item_eval_pred = tf.matmul(prev_item_emb, self._item_embeddings, transpose_b=True)
		#item_eval_pred = tf.matmul(test_item_emb, self._item_embeddings, transpose_b=True)
		eval_pred = user_eval_pred + item_eval_pred
		
		user_item_pred, item_item_pred = self.predict(eval_user, prev_item)
		
		test_res = tf.cast((user_item_pred + item_item_pred) > eval_pred, tf.int32)
		test_rank_per_user = tf.reduce_sum(test_res, axis=1)
		test_rank = tf.cast(test_rank_per_user <= self.top_k, tf.int32)

		# calculate HR & MRR
		self._test_hr = tf.reduce_sum(test_rank)
		self._test_mrr = tf.reduce_sum(tf.math.reciprocal_no_nan(tf.cast(tf.multiply(test_rank, test_rank_per_user), dtype=tf.float32)))

	def eval(self):
		test_hr, test_mrr = 0.0, 0.0

		user_ind, users, prev_items, test_items = self.get_eval_batch()

		feed_dict = {
			self._eval_user: users,
			self._prev_item: prev_items,
			self._test_item: test_items
		}

		[test_hr, test_mrr] = self._sess.run(
			[self._test_hr, self._test_mrr],
			feed_dict=feed_dict)

		avg_test_hr = test_hr / user_ind
		avg_test_mrr = test_mrr / user_ind
		
		print(' ---------------------------------------------------------------------------------')
		print(' Eval ')
		print("HR: ", avg_test_hr, "	MRR:", avg_test_mrr)
		print(' ---------------------------------------------------------------------------------')
		
		return avg_test_hr, avg_test_mrr

	def get_eval_batch(self):
		users, prev_items, test_items = list(), list(), list()

		users = np.arange(self.user_num)
		user_ind = 0

		for user in users:
			prev_items.append(self.user_train[user][-1][1])
			test_items.append(self.user_test[user][0][1])
			user_ind += 1

		return user_ind, users, prev_items, test_items

