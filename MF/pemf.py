import numpy as np
import random
import math
import tensorflow as tf
from tensorflow import keras
import time

from params import Params

class PEMF(object):
	def __init__(self, dataset, params, sess):
		# dataset
		[self.user_train, self.user_test, self.user_num, self.item_num] = dataset
		self._sess = sess
		
		# hyper-parameter
		self.epsilon = params.epsilon
		self.emb_dim = params.emb_dim
		self.proj_dim = params.proj_dim
		self.reg = params.reg
		self.epoches = params.epoches
		self.learning_rate = params.learning_rate

		# evaluation
		self.top_k = params.top_k
		self.best_hr = tf.compat.v1.Variable(-1.0, dtype=tf.float32, trainable=False, name='best_hr')
		self.best_mrr = tf.compat.v1.Variable(-1.0, dtype=tf.float32, trainable=False, name='best_mrr')
		self.best_iter = tf.compat.v1.Variable(0, dtype=tf.int32, trainable=False, name='best_iter')

		with tf.name_scope('training'):
			self._total_loss = None
			self._batch_loss = None
			self._reg_loss = None
			self._train_op = None

		# build graph
		self.build_graph()
		self.build_eval_graph()

	def build_graph(self):
		# Nodes in the graph which are used to run/feed/fetch
		with tf.name_scope("input"):
			cur_user = tf.compat.v1.placeholder(tf.int32, [None], name='cur_user')
			user_item_matrix = tf.compat.v1.placeholder(tf.int32, [None, self.item_num], name='user_item_matrix')
			proj_dim_ind = tf.compat.v1.placeholder(tf.int32, name='proj_dim_ind')
			emb_dim_ind = tf.compat.v1.placeholder(tf.int32, name='emb_dim_ind')
			grad_matrix_ind = tf.compat.v1.placeholder(tf.int32, name='grad_matrix_ind')
			mask_indices = tf.compat.v1.placeholder(tf.int32, [None, 2], name='mask_indices')
			self._cur_user, self._proj_dim_ind, self._emb_dim_ind, self._grad_matrix_ind, self._user_item_matrix, self._mask_indices = \
				cur_user, proj_dim_ind, emb_dim_ind, grad_matrix_ind, user_item_matrix, mask_indices

			# build mask matrix
			rating_num = tf.shape(mask_indices)[0]
			dense_shape = tf.compat.v1.constant([self.user_num, self.item_num], dtype=tf.int64)
			values = tf.ones([rating_num])
			mask = tf.sparse.SparseTensor(indices=tf.dtypes.cast(self._mask_indices, tf.int64),
										  values=values,
										  dense_shape=dense_shape)
			dense_mask = tf.sparse.to_dense(mask, default_value=0, validate_indices=False)

		# get positive/negative item score
		pred = self.forward(cur_user)

		batch_loss, reg_loss, grads_per_user = self.loss(pred, cur_user, emb_dim_ind, dense_mask)

		self._batch_loss, self._reg_loss = batch_loss / tf.cast(rating_num, dtype=tf.float32), reg_loss
		self._total_loss = batch_loss + reg_loss
		self._grads_per_user = grads_per_user

		self.optimize(self._total_loss, grads_per_user, grad_matrix_ind)

	def forward(self, cur_user):
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

			projection_matrix = tf.random.normal(
				name = 'projection_matrix',
				shape = [self.proj_dim, self.item_num],
				mean = 0.0,
				stddev = 1.0 / tf.sqrt(float(self.proj_dim)))
			self._projection_matrix = projection_matrix

			return self.predict(cur_user)

	def predict(self, cur_user):
		# shape [cur_batch_size, emb_dim]
		cur_batch_size = tf.shape(cur_user)[0]
		cur_user_emb = tf.gather(self._user_embeddings, cur_user)
		pred = tf.matmul(cur_user_emb, self._item_embeddings, transpose_b=True)

		# shape [cur_batch_size, target_item_num]
		return pred

	def loss(self, pred, cur_user, emb_dim_ind, dense_mask):
		user_emb_nd = tf.expand_dims(tf.gather_nd(self._user_embeddings, emb_dim_ind), axis=1)
		cur_user_emb = tf.gather(self._user_embeddings, cur_user)
		cur_user_rating = tf.dtypes.cast(tf.gather(self._user_item_matrix, cur_user), tf.float32)
		related_params = tf.concat([cur_user_emb, self._item_embeddings], 0)

		error = tf.multiply((cur_user_rating - pred), dense_mask)
		batch_loss = tf.nn.l2_loss(error)
		reg_loss = tf.nn.l2_loss(related_params) * self.reg
		grads_per_user = tf.multiply(error, user_emb_nd)

		return batch_loss, reg_loss, grads_per_user

	def optimize(self, loss, grads_per_user, grad_matrix_ind):
		# user embedding update
		user_grads = tf.gradients(loss, [self._user_embeddings])

		# gradient perturbation
		proj_emb = tf.gather(self._projection_matrix, self._proj_dim_ind)
		proj_grads = tf.reduce_sum(tf.multiply(proj_emb, grads_per_user), axis=1)
		proj_grads = tf.clip_by_value(proj_grads, -1.0, 1.0)
		proj_grads = (proj_grads * (tf.exp(self.epsilon / self.epoches) - 1) + (tf.exp(self.epsilon / self.epoches) + 1)) / (2 * tf.exp(self.epsilon / self.epoches) + 2)

		prob = tf.random.uniform([self.user_num], minval=0.0, maxval=1.0)
		proj_grads = (tf.cast(proj_grads < prob, tf.float32) - 0.5) * 2 * \
			self.proj_dim * self.emb_dim * (tf.exp(self.epsilon / self.epoches) + 1) / (tf.exp(self.epsilon / self.epoches) - 1)
			
		# build mask matrix
		proj_grads = tf.scatter_nd(grad_matrix_ind, proj_grads, [self.proj_dim, self.emb_dim]) / self.user_num
		pseudo_inv = tf.matmul(tf.transpose(self._projection_matrix), tf.linalg.inv(tf.matmul(self._projection_matrix, self._projection_matrix, transpose_b=True)))
		proj_grads = tf.matmul(pseudo_inv, proj_grads)

		# apply update
		update_user = tf.compat.v1.assign(self._user_embeddings, self._user_embeddings - tf.scalar_mul(self.learning_rate, user_grads[0]))
		self._update_user = update_user
		
		update_item = tf.compat.v1.assign(self._item_embeddings, self._item_embeddings - proj_grads * self.learning_rate)
		self._update_item = update_item

	def train(self, is_eval):
		start = time.time()
		self._sess.run(tf.compat.v1.global_variables_initializer())
		user_ind, user_item_matrix, mask_indices = self.get_full_matrix()

		for epoch in range(self.epoches):
			users, proj_dim_ind, emb_dim_ind, grad_matrix_ind = self.get_batch()

			feed_dict = {
				self._cur_user: users,
				self._proj_dim_ind: proj_dim_ind,
				self._emb_dim_ind: emb_dim_ind,
				self._grad_matrix_ind: grad_matrix_ind,
				self._user_item_matrix: user_item_matrix,
				self._mask_indices: mask_indices
			}

			[_, _, batch_loss, reg_loss] = self._sess.run(
				[self._update_user, self._update_item, self._batch_loss, self._reg_loss],
				feed_dict=feed_dict)

			print("step: ", epoch + 1, "	total_loss", batch_loss + reg_loss, " batch_loss: ", batch_loss, " reg_loss :", reg_loss)

			if is_eval == True:
				avg_test_hr, avg_test_mrr = self.eval()
				best_hr, best_mrr, best_iter = self.best_hr.eval(), self.best_mrr.eval(), self.best_iter.eval()
				if avg_test_hr >= best_hr:
					self._sess.run([
						tf.compat.v1.assign(self.best_hr, avg_test_hr),
						tf.compat.v1.assign(self.best_mrr, avg_test_mrr),
						tf.compat.v1.assign(self.best_iter, epoch)])
		
	def get_full_matrix(self):
		mask_indices = list()
		user_item_matrix = np.zeros([self.user_num, self.item_num], dtype=np.int32)

		user_ind = 0

		for user in range(self.user_num):
			for i in range(len(self.user_train[user])):
				user_item_matrix[user][self.user_train[user][i][1]]+=1
				mask_indices.append([user_ind, self.user_train[user][i][1]])
			user_ind+=1

		return user_ind, user_item_matrix, mask_indices

	def get_batch(self):
		proj_dim_ind, emb_dim_ind, grad_matrix_ind = list(), list(), list()
		user_ind = 0

		users = np.arange(self.user_num)

		for user in users:
			s = np.random.randint(0, self.proj_dim)
			l = np.random.randint(0, self.emb_dim)

			proj_dim_ind.append(s)
			emb_dim_ind.append([user_ind, l])
			grad_matrix_ind.append([s, l])
			user_ind += 1

		return users, proj_dim_ind, emb_dim_ind, grad_matrix_ind
############################################################################################################################
	def build_eval_graph(self):
		self._eval_user = tf.compat.v1.placeholder(dtype=tf.int32, name='eval_user')
		self._test_item = tf.compat.v1.placeholder(dtype=tf.int32, name='test_item')

		eval_user_emb = tf.gather(self._user_embeddings, self._eval_user)
		test_item_emb = tf.gather(self._item_embeddings, self._test_item)
		eval_pred = tf.reduce_sum(tf.multiply(eval_user_emb, test_item_emb), axis=1)
		eval_pred = tf.tile(tf.expand_dims(eval_pred, 1), [1, self.item_num])
		pred = self.predict(self._eval_user)

		test_res = tf.cast(pred > eval_pred, tf.int32)
		test_rank_per_user = tf.reduce_sum(test_res, axis=1)
		test_rank = tf.cast(test_rank_per_user <= self.top_k, tf.int32)

		# calculate HR
		self._test_hr = tf.reduce_sum(test_rank)
		self._test_mrr = tf.reduce_sum(tf.math.reciprocal_no_nan(tf.cast(tf.multiply(test_rank, test_rank_per_user), dtype=tf.float32)))

	def eval(self):
		test_hr, test_mrr = 0.0, 0.0
		
		user_ind, users, test_items = self.get_eval_batch()

		feed_dict = {
			self._eval_user: users,
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
		users, test_items = list(), list()

		users = np.arange(self.user_num)
		user_ind = 0

		for user in users:
			test_items.append(self.user_test[user][0][1])
			user_ind += 1

		return user_ind, users, test_items

