# Matrix Factorization & Privacy Enhanced Matrix Factorization
import numpy as np
import tensorflow as tf

from params import Params
from mf import MF
from pemf import PEMF
from pemf_movie import PEMF_MOVIE

def main(_):
	params = Params()
	dataset = np.load('../data/' + params.data_path + '.npy', allow_pickle=True)
	[user_train, user_test, user_num, item_num] = dataset

	config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth = True

	is_eval = True
	
	print("Loading", params.data_path, '.npy...')
	print("# users : ", user_num)
	print("# items : ", item_num)
	
	with tf.Graph().as_default():
		sess = tf.compat.v1.Session(config=config)
		with sess.as_default():
			model = MF(dataset, params, sess)
			#model = PEMF(dataset, params, sess)
			model.train(is_eval)

if __name__ == "__main__":
	tf.compat.v1.app.run()
