import numpy as np
import math
import tensorflow as tf
import time

from params import Params

class ICF(object):
	def __init__(self, dataset, params, sess):
		# dataset
		[self.user_train, self.user_test, self.user_num, self.item_num] = dataset
		self._sess = sess
		
		# parameter
		self.epsilon = params.epsilon
		self.prob = math.exp(self.epsilon) / (1 + math.exp(self.epsilon))
		self.alpha = params.alpha
		self.batch_size = params.batch_size
		self.joint_batch = params.joint_batch

		# evaluation paramater
		self.top_k = params.top_k
		
		# input
		self._noisy_user_item_matrix = None
		self._similarity_matrix = None

		# build graph
		self.build_frequency_graph()
		self.build_joint_frequency_graph()
		self.build_pred_graph()

	def build_frequency_graph(self):
		with tf.name_scope('frequency'):
			pair_indices = tf.compat.v1.placeholder(dtype=tf.int32, name="pair_indices")
			self._pair_indices = pair_indices
		
		self.get_frequency(pair_indices)

	def build_joint_frequency_graph(self):
		with tf.name_scope('joint_frequency'):
			cur_batch = tf.compat.v1.placeholder(dtype=tf.int32, name="cur_batch")
			noisy_user_item_matrix = tf.compat.v1.placeholder(dtype=tf.int32)
			self._cur_batch, self_noisy_user_item_matrix = cur_batch, noisy_user_item_matrix

		pos_neg_freq, neg_pos_freq, neg_neg_freq = self.get_joint_frequency(cur_batch)
		self._pos_neg_freq, self._neg_pos_freq, self._neg_neg_freq = pos_neg_freq, neg_pos_freq, neg_neg_freq
		
	def build_pred_graph(self):
		with tf.name_scope('predict'):
			similarity_matrix = tf.compat.v1.placeholder(dtype=tf.float32, name="similarity_matrix")
			noisy_user_item_mat = tf.compat.v1.placeholder(dtype=tf.float32)
			self._similarity_matrix, self._noisy_user_item_mat = similarity_matrix, noisy_user_item_mat
		
		pred = self.predict(similarity_matrix, noisy_user_item_mat)
		self._pred = pred
		
	def get_frequency(self, pair_indices):
		# calculate
		pair_num = tf.shape(pair_indices)[0]

		# construct user_item_matrix
		values = tf.ones([pair_num])
		sparse_user_item_matrix = tf.sparse.SparseTensor(indices=tf.dtypes.cast(pair_indices, tf.int64),
									values=values,
									dense_shape=[self.user_num, self.item_num])
		user_item_matrix = tf.sparse.to_dense(sparse_user_item_matrix, default_value=0, validate_indices=False)
		self._user_item_matrix = user_item_matrix

		# randomized response
		prob_matrix = tf.random.uniform([self.user_num, self.item_num], 0.0, 1.0)
		noisy_user_item_matrix = tf.cast(prob_matrix < self.prob, tf.float32) * user_item_matrix
		self._noisy_user_item_matrix = noisy_user_item_matrix
		
		# frequency estimation
		noisy_freq = tf.reduce_sum(noisy_user_item_matrix, axis=0)
		self._noisy_freq = noisy_freq

	def get_joint_frequency(self, cur_batch):
		eval_item = tf.expand_dims(tf.slice(self._noisy_user_item_matrix, [0, cur_batch], [self.user_num, self.joint_batch]), 2)
		eval_item = tf.tile(eval_item, [1, 1, self.item_num])
		inv_eval_item = 1 - eval_item

		pos_neg_freq = tf.reduce_sum(tf.multiply(eval_item, tf.expand_dims(1-self._noisy_user_item_matrix, 1)), 0)
		neg_pos_freq = tf.reduce_sum(tf.multiply(inv_eval_item, tf.expand_dims(self._noisy_user_item_matrix, 1)), 0)
		neg_neg_freq = tf.reduce_sum(tf.multiply(inv_eval_item, tf.expand_dims(1-self._noisy_user_item_matrix, 1)), 0)

		return pos_neg_freq, neg_pos_freq, neg_neg_freq

	def predict(self, similarity_matrix, noisy_user_item_mat):
		pred = tf.matmul(tf.cast(noisy_user_item_mat, tf.float32), similarity_matrix, transpose_b=True)
		similarity_sum = tf.tile(tf.expand_dims(tf.reduce_sum(similarity_matrix, 1), 0), [self.user_num, 1])
		pred /= similarity_sum
		
		return pred

	def process(self):
		start = time.time()
		self._sess.run(tf.compat.v1.global_variables_initializer())
		
		# frequency estimation input
		estimated_freq = np.zeros(self.item_num)

		user_ind, users, pair_indices = self.get_batch()
		
		feed_dict = {
			self._pair_indices: pair_indices
		}

		[noisy_user_item_matrix, noisy_freq] = self._sess.run(
			[self._noisy_user_item_matrix, self._noisy_freq],
			feed_dict=feed_dict)

		
		estimated_freq = (noisy_freq - self.user_num * (1.0 - self.prob)) / (2.0 * self.prob - 1.0)
		estimated_freq[estimated_freq < 0] = 1

		# joint frequency estimation input
		start = time.time()
		pos_neg_matrix = np.zeros([self.item_num, self.item_num])
		neg_pos_matrix = np.zeros([self.item_num, self.item_num])
		neg_neg_matrix = np.zeros([self.item_num, self.item_num])
		
		num_batch = int(self.item_num / self.joint_batch)
		
		for b_i in range(num_batch):
			feed_dict = {
				self._cur_batch: b_i,
				self._noisy_user_item_matrix: noisy_user_item_matrix
			}

			[pos_neg_freq, neg_pos_freq, neg_neg_freq] = self._sess.run(
				[self._pos_neg_freq, self._neg_pos_freq, self._neg_neg_freq],
				feed_dict=feed_dict)

			pos_neg_matrix[b_i * self.joint_batch:(b_i + 1) * self.joint_batch,:] += pos_neg_freq
			neg_pos_matrix[b_i * self.joint_batch:(b_i + 1) * self.joint_batch,:] += neg_pos_freq
			neg_neg_matrix[b_i * self.joint_batch:(b_i + 1) * self.joint_batch,:] += neg_neg_freq	
		
		# get similarity matrix
		prob_square = math.pow(self.prob, 2)
		term1 = prob_square * self.user_num
		term2 = self.prob * (pos_neg_matrix + neg_pos_matrix)
		term3 = (2 * self.prob - 1) * neg_neg_matrix
		similarity_matrix = (term1 - term2 - term3) / (4*prob_square - 4*self.prob + 1)
		
		for i in range(self.item_num):
			for j in range(self.item_num):
				similarity_matrix[i][j] = similarity_matrix[i][j] / estimated_freq[i] / math.pow(estimated_freq[j], self.alpha)
		
		# evaluation
		correct_user = 0
		mrr = 0

		feed_dict = {
			self._similarity_matrix: similarity_matrix,
			self._noisy_user_item_mat: noisy_user_item_matrix
		}

		[pred] = self._sess.run(
			[self._pred],
			feed_dict=feed_dict)

		for user in range(self.user_num):
			result = self.eval(user, pred)
			correct_user += result[0]
			mrr += result[1]
		
	def eval(self, user, pred):
		test_poi = self.user_test[user][0][1]
		candidate = pred[user,:]

		top_k_loc_id = np.zeros(self.top_k, np.int32)
		for i in range(self.top_k):
			idx = np.argmax(candidate)
			top_k_loc_id[i] = idx
			candidate[idx] = 0

		if test_poi in top_k_loc_id:
			for i in range(self.top_k):
				if test_poi == top_k_loc_id[i]:
					return [1, 1.0 / (i + 1)]
		else:
			return [0, 0]

	def get_batch(self):
		pair_indices = list()

		users = np.arange(self.user_num)

		user_ind = 0

		# sample evaluation batch from dataset
		for user in users:
			for l in range(len(self.user_train[user])):
				pair_indices.append([user_ind, self.user_train[user][l][1]])

			user_ind += 1

		return user_ind, users, pair_indices
