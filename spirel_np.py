import numpy as np
import random
import math
import tensorflow as tf
import time

from numpy import array
from params import Params

class SPIREL_NP(object):
	def __init__(self, dataset, params, sess):
		# dataset & session
		[self.user_train, self.user_test, self.user_num, self.item_num] = dataset
		self.output_path = params.np_output_path
		self._sess = sess
		
		# hyper-parameter
		self.emb_dim = params.emb_dim  # size of latent dimensions
		self.epoches = params.epoches  # number of max iteration
		self.learning_rate = params.learning_rate # user latent learning rate
		self.user_reg = params.user_reg # regularization parameter
		self.item_reg = params.item_reg # regularization parameter
		
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
		self.build_graph()
		self.build_eval_graph()
	
	def build_graph(self):
		with tf.name_scope("matrix_input"):
			user_item_matrix = tf.compat.v1.placeholder(tf.float32, name='user_item_matrix')
			item_item_matrix = tf.compat.v1.placeholder(tf.float32, name='item_item_matrix')
			user_item_indices = tf.compat.v1.placeholder(tf.float32, name='user_item_indices')
			item_item_indices = tf.compat.v1.placeholder(tf.float32, name='item_item_indices')
			self._user_item_matrix, self._item_item_matrix, self._user_item_indices, self._item_item_indices = \
				user_item_matrix, item_item_matrix, user_item_indices, item_item_indices
			
		with tf.name_scope("batch_input"):	
			cur_user = tf.compat.v1.placeholder(tf.int32, name='cur_user')
			cur_item = tf.compat.v1.placeholder(tf.int32, name='cur_item')
			self._cur_user, self._cur_item = cur_user, cur_item
		
		# calculate
		user_item_pair = tf.shape(user_item_indices)[0]
		item_item_pair = tf.shape(item_item_indices)[0]
		
		user_item_pred, item_item_pred = self.forward(cur_user, cur_item)
		
		user_item_loss, item_item_loss, user_item_loss_l2, item_item_loss_l2, batch_loss, reg_loss = \
			self.loss(cur_user, cur_item, user_item_pred, item_item_pred, user_item_pair, item_item_pair)
		self._user_item_loss, self._item_item_loss, self._user_item_loss_l2, self._item_item_loss_l2, self._batch_loss, self._reg_loss = \
			user_item_loss, item_item_loss, user_item_loss_l2, item_item_loss_l2, batch_loss, reg_loss
		#(batch_loss / tf.cast(pair_num, dtype=tf.float32))
		self.optimize(cur_user, batch_loss, reg_loss)
			
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
	
	def optimize(self, cur_user, batch_loss, reg_loss):	
		# calculate
		cur_batch_size = tf.shape(cur_user)[0]
	
		# user embedding update (ALS)
		inv_als_term = tf.linalg.inv(tf.matmul(self._item_embeddings, self._item_embeddings, transpose_a=True) + \
							tf.multiply(float(self.user_reg), tf.eye(self.emb_dim)))
		user_term = tf.matmul(tf.cast(self._user_item_matrix, tf.float32), tf.matmul(self._item_embeddings, inv_als_term))
		
		update_user = tf.compat.v1.assign(self._user_embeddings, user_term)
		self._update_user = update_user
		
		# item embedding update (Adam)
		optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta_1, beta2=self.beta_2, epsilon=self.adam_eps)
		gvs = optimizer.compute_gradients(batch_loss + reg_loss, [self._item_embeddings])
		train_op = optimizer.apply_gradients(gvs, global_step=self._global_step)
		self._train_op = train_op
		
	def train(self, is_eval):
		start = time.time()
		# input initialize
		self._sess.run(tf.compat.v1.global_variables_initializer())
		user_item_matrix, item_item_matrix, user_item_indices, item_item_indices = self.get_matrix()
		item_item_matrix = np.log10(item_item_matrix)
		item_item_matrix = 1 + (1.0 / (1.0 + np.exp(-item_item_matrix)))
				
		# learning
		for epoch in range(self.epoches):
			users, items = self.get_batch()
			
			feed_dict = {
				self._user_item_matrix: user_item_matrix,
				self._item_item_matrix: item_item_matrix,
				self._user_item_indices: user_item_indices,
				self._item_item_indices: item_item_indices,
				self._cur_user: users,
				self._cur_item: items
			}
			
			[_, _, user_item_loss_l2, item_item_loss_l2, batch_loss, reg_loss] = self._sess.run(
				[self._update_user, self._train_op, self._user_item_loss_l2, self._item_item_loss_l2, self._batch_loss, self._reg_loss],
				feed_dict=feed_dict)
			
			print("step: ", epoch + 1, " total_loss: ", batch_loss + reg_loss, " reg_loss :", reg_loss, " user_loss: " , user_item_loss_l2, " item_loss :", item_item_loss_l2)
			
			# evaluation
			if is_eval == True:
				avg_test_hr, avg_test_mrr = self.eval()
				best_hr, best_mrr, best_iter = self.best_hr.eval(), self.best_mrr.eval(), self.best_iter.eval()

				if avg_test_hr > best_hr:
					self._sess.run([
						tf.compat.v1.assign(self.best_hr, avg_test_hr),
						tf.compat.v1.assign(self.best_mrr, avg_test_mrr),
						tf.compat.v1.assign(self.best_iter, epoch)])
						
		# result
		print("time :", time.time() - start)
		f = open('../result/SPIREL_NP/' + self.output_path + '.txt', 'a')
		f.write('%f	%f\n' %(round(self.best_hr.eval(), 4), round(self.best_mrr.eval(), 4)))
		print('best_hr:', round(self.best_hr.eval(), 4))
		print('best_mrr:', round(self.best_mrr.eval(), 4))
		f.close()
		
	def get_matrix(self):
		user_item_indices, item_item_indices = list(), list()
		user_item_matrix = np.zeros([self.user_num, self.item_num], dtype=np.int32)
		item_item_matrix = np.zeros([self.item_num, self.item_num], dtype=np.int32)

		for user in range(self.user_num):
			for i in range(len(self.user_train[user])):
				user_item_matrix[user][self.user_train[user][i][1]] += 1
				user_item_indices.append([user, self.user_train[user][i][1]])
				
				if i != len(self.user_train[user]) - 1:
					item_item_matrix[self.user_train[user][i][1]][self.user_train[user][i+1][1]] += 1
					item_item_indices.append([self.user_train[user][i][1], self.user_train[user][i+1][1]])

		return user_item_matrix, item_item_matrix, user_item_indices, item_item_indices
	
	def get_batch(self): 		
		users = np.arange(self.user_num)
		items = np.arange(self.item_num)

		return users, items
		
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
		#item_eval_pred = tf.matmul(self._item_embeddings, test_item_emb, transpose_b=True)
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

